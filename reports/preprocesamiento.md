# Preprocesamiento de datos — MiBici (Fase 1)

## 1. Objetivo

El objetivo del preprocesamiento es transformar los registros individuales de viajes de MiBici en un conjunto de datos organizado por estación y hora, incorporando variables que permitan desarrollar posteriormente modelos de regresión para pronosticar la demanda de bicicletas.

**Entrada:** `data/interim/viajes_limpios.parquet`

**Salida:** `data/processed/estacion_hora.parquet`


## 2. Agregación por estación y hora

Se agruparon los viajes utilizando el identificador de la estación y la fecha y hora correspondiente. Para cada combinación se calcularon las salidas y llegadas de bicicletas.

A partir de 4,354,416 viajes limpios se generaron 3,446,117 registros por estación y hora. Se conservaron las 4,354,416 salidas, mientras que las llegadas sumaron 4,354,400 debido a la exclusión de 16 llegadas posteriores al cierre del periodo.

También se incorporaron 1,519,262 registros sin movimientos mediante la generación de una malla horaria por estación. Esta comienza en la primera actividad observada y, para las estaciones con actividad reciente, se extiende hasta el 30 de septiembre de 2026 a las 23:00.

En la segunda versión del preprocesamiento se eliminaron 15 registros estación-hora del 1 de octubre de 2026, correspondientes a 16 llegadas. Además, se agregaron 555 horas sin movimientos en 302 estaciones para completar el periodo establecido.

Se decidió no extender las cuatro estaciones cuya última actividad ocurrió antes del 29 de septiembre, debido a la posibilidad de que hubieran dejado de operar. Entre ellas se encuentra la estación 465, cuya actividad terminó el 28 de septiembre a las 22:00, solamente 49 horas antes del cierre del periodo. Este caso requiere especial atención, ya que también podría deberse a una baja demanda y no necesariamente al cierre de la estación.

**Limitación:** la ausencia de viajes no permite distinguir completamente entre periodos de demanda cero y estaciones temporalmente fuera de operación. Para confirmar estos casos sería necesario contar con información adicional sobre el estado operativo de las estaciones.

## 3. Ingeniería de variables

Se incorporaron las siguientes características:

- **Temporales:** hora, día de la semana, mes y fin de semana.
- **Festivos:** variable binaria que identifica los días de descanso obligatorio considerados dentro del periodo analizado.
- **Históricas:** lags de salidas y llegadas correspondientes a 1, 24 y 168 horas anteriores.
- **Cíclicas:** transformaciones mediante seno y coseno para representar adecuadamente la periodicidad de horas, días y meses.

Los lags de 1, 24 y 168 horas se seleccionaron inicialmente para representar el comportamiento reciente, diario y semanal de la demanda. Su utilidad predictiva deberá contrastarse posteriormente con las conclusiones del EDA y los resultados del modelado.

## 4. Normalización y tratamiento de valores faltantes

Se utilizó `StandardScaler` de Scikit-learn para normalizar las seis variables históricas.

El escalador fue ajustado exclusivamente con registros anteriores al 1 de agosto de 2026, evitando incorporar estadísticas del periodo posterior durante el ajuste.

Los valores faltantes en los lags se imputaron mediante medianas calculadas con datos de entrenamiento. Adicionalmente, se incorporó `historial_incompleto` para identificar registros que no cuentan con suficiente información histórica.

El escalador y las medianas se almacenaron en `data/processed/escalador_lags.pkl`, permitiendo reutilizar las transformaciones.

El corte temporal se utiliza para ajustar los transformadores; la separación física en conjuntos de entrenamiento, validación y prueba no forma parte de esta implementación.

## 5. Codificación de estaciones

Debido a que `estacion_id` representa una categoría y no una magnitud numérica, se utilizó `OneHotEncoder` con `handle_unknown="ignore"` y salida dispersa.

El codificador se ajustó exclusivamente con los datos del periodo de entrenamiento, identificando 455 estaciones, y se guardó en `data/processed/codificador_estaciones.pkl`.

La codificación One-Hot no se materializó dentro del archivo Parquet. El transformador quedó preparado para aplicarse durante la construcción de la matriz de características del modelo, evitando incorporar cientos de columnas adicionales en esta fase.

Se identificaron 16 estaciones que aparecen únicamente después del corte de entrenamiento del 1 de agosto de 2026. Debido a que el codificador se ajustó exclusivamente con los datos anteriores a esa fecha, estas estaciones no forman parte de las categorías aprendidas y se representan mediante vectores de ceros al aplicar One-Hot Encoding.

## 6. Reducción de dimensionalidad

No se aplicó reducción de dimensionalidad durante la Fase 1.

Se decidió conservar las variables temporales e históricas para evaluar posteriormente su contribución al pronóstico. Además, la representación dispersa de OneHotEncoder permite manejar las estaciones sin construir una matriz densa.

La posible eliminación o reducción de características se evaluará después del EDA y de las pruebas de entrenamiento, cuando exista evidencia para determinar cuáles aportan información útil.


## 7. Versionado y trazabilidad

El preprocesamiento se registró en el experimento `mibici-fase1` de MLflow, generando la nueva versión `mibici_processed_v2` con el contexto `processed`, sin sobrescribir `mibici_processed_v1`, que se conserva como versión anterior.

Para garantizar la trazabilidad de los datos, se incorporó el registro de `mibici_interim_v1` con el contexto `interim`, permitiendo identificar el conjunto de datos limpios utilizado como entrada del preprocesamiento.

La ejecución `preprocesamiento_v2` contiene ambos datasets, sus identificadores *digest*, parámetros del procesamiento, esquema de datos y los artefactos generados, incluyendo el archivo Parquet procesado, el escalador y el codificador de estaciones.

Además, se registraron los tags `git_commit`, `git_cambios_sin_commit` y `fuente_datos`, siguiendo el patrón de versionado implementado en `mibici/cleaning.py`. Esto permite relacionar los resultados con la versión del código fuente y el origen de los datos utilizados.

Los datos y artefactos locales no se suben a GitHub; únicamente se versionan los scripts, las pruebas automatizadas y la documentación técnica.

## 8. Validación

La implementación fue revisada mediante Ruff y pruebas automatizadas con Pytest.

Se obtuvieron nueve pruebas aprobadas: cinco correspondientes a limpieza y cuatro al preprocesamiento.

Las pruebas de preprocesamiento verifican la agregación de viajes, los registros sin movimientos, la normalización sin utilizar estadísticas del periodo posterior y el manejo de estaciones desconocidas durante la codificación.

## 9. Consideraciones para el modelado

Las columnas `salidas` y `llegadas` representan los valores que se pueden pronosticar y no deben utilizarse como predictores de la misma hora.

Las variables históricas solo deberán utilizar información disponible en el momento de generar cada predicción.

Queda pendiente contrastar las características seleccionadas con los resultados del EDA, analizar los periodos operativos de las estaciones y evaluar el desempeño de diferentes configuraciones de variables.
