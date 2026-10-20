# Preprocesamiento de datos — MiBici (Fase 1)

## 1. Objetivo

El objetivo del preprocesamiento es transformar los registros individuales de viajes de MiBici en un conjunto de datos organizado por estación y hora, incorporando variables que permitan desarrollar posteriormente modelos de regresión para pronosticar la demanda de bicicletas.

**Entrada:** `data/interim/viajes_limpios.parquet`

**Salida:** `data/processed/estacion_hora.parquet`

## 2. Agregación por estación y hora

Se agruparon los viajes utilizando el identificador de la estación y la fecha y hora correspondiente. Para cada combinación se calcularon las salidas y llegadas de bicicletas.

A partir de 4,354,416 viajes limpios se generaron 3,445,577 registros por estación y hora, conservando los totales originales de salidas y llegadas.

También se incorporaron 1,518,707 registros sin movimientos, mediante la generación de intervalos horarios entre la primera y última actividad observada de cada estación.

**Limitación:** estos intervalos no permiten distinguir completamente entre horas de demanda cero y periodos en los que una estación estuvo temporalmente fuera de operación. La identificación de periodos operativos queda pendiente de información adicional.

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

## 6. Reducción de dimensionalidad

No se aplicó reducción de dimensionalidad durante la Fase 1.

Se decidió conservar las variables temporales e históricas para evaluar posteriormente su contribución al pronóstico. Además, la representación dispersa de OneHotEncoder permite manejar las estaciones sin construir una matriz densa.

La posible eliminación o reducción de características se evaluará después del EDA y de las pruebas de entrenamiento, cuando exista evidencia para determinar cuáles aportan información útil.

## 7. Versionado y trazabilidad

El dataset procesado se registró en el experimento `mibici-fase1` de MLflow con el nombre `mibici_processed_v1` y el contexto `processed`.

El registro incluye parámetros del preprocesamiento, esquema, digest del dataset, archivo Parquet y transformadores utilizados.

Se incorporaron los tags `git_commit`, `git_cambios_sin_commit` y `fuente_datos`, siguiendo el patrón de versionado implementado en `mibici/cleaning.py`.

Los datos y artefactos locales no se suben a GitHub; únicamente se versionan los scripts, pruebas y documentación.

## 8. Validación

La implementación fue revisada mediante Ruff y pruebas automatizadas con Pytest.

Se obtuvieron ocho pruebas aprobadas: cinco correspondientes a limpieza y tres al preprocesamiento.

Las pruebas de preprocesamiento verifican la agregación de viajes, los registros sin movimientos, la normalización sin utilizar estadísticas del periodo posterior y el manejo de estaciones desconocidas durante la codificación.

## 9. Consideraciones para el modelado

Las columnas `salidas` y `llegadas` representan los valores que se pueden pronosticar y no deben utilizarse como predictores de la misma hora.

Las variables históricas solo deberán utilizar información disponible en el momento de generar cada predicción.

Queda pendiente contrastar las características seleccionadas con los resultados del EDA, analizar los periodos operativos de las estaciones y evaluar el desempeño de diferentes configuraciones de variables.
