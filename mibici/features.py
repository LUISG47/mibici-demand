"""Agregación de viajes de MiBici por estación y hora."""

from math import cos, pi, sin
from pathlib import Path
import pickle

from loguru import logger
import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mibici.cleaning import estado_git
from mibici.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    cargar_parametros,
    configurar_mlflow,
)
from mibici.dataset import URL_DATOS_ABIERTOS
from mibici.modeling.tracking import configurar_mlflow as configurar_mlflow_modelado

app = typer.Typer()

# Fechas utilizadas en el preprocesamiento
FECHA_CORTE = pd.Timestamp("2026-08-01")
FIN_PERIODO = pd.Timestamp("2026-09-30 23:00:00")
INICIO_EXTENSION = pd.Timestamp("2026-09-29 00:00:00")


def contar_eventos(estaciones, fechas, nombre):
    """Cuenta eventos por estación y hora."""

    eventos = pd.DataFrame(
        {
            "estacion_id": estaciones,
            "fecha_hora": fechas.dt.floor("h"),
        }
    )

    return eventos.groupby(["estacion_id", "fecha_hora"]).size().rename(nombre)


def generar_dataset_estacion_hora(viajes, rezagos=None):
    """Genera los conteos horarios de salidas y llegadas."""

    if rezagos is None:
        rezagos = cargar_parametros()["features"]["rezagos"]

    columnas = [
        "Origen_Id",
        "Destino_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
    ]

    if viajes[columnas].isna().any().any():
        raise ValueError("Existen valores nulos en las columnas necesarias.")

    # Contar las salidas según la estación y hora de inicio
    salidas = contar_eventos(
        viajes["Origen_Id"],
        viajes["Inicio_del_viaje"],
        "salidas",
    )

    # Contar las llegadas según la estación y hora de fin
    llegadas = contar_eventos(
        viajes["Destino_Id"],
        viajes["Fin_del_viaje"],
        "llegadas",
    )

    # Unir ambos conteos
    conteos = pd.concat([salidas, llegadas], axis=1).fillna(0).astype("int32").reset_index()

    # Excluir movimientos registrados fuera del periodo
    conteos = conteos[conteos["fecha_hora"] <= FIN_PERIODO].copy()

    # Incorporar horas sin viajes para cada estación
    bloques = []

    for estacion, grupo in conteos.groupby("estacion_id"):
        primera_actividad = grupo["fecha_hora"].min()
        ultima_actividad = grupo["fecha_hora"].max()

        # Extender las estaciones con actividad desde el 29 de septiembre
        # Las que dejaron de operar antes conservan su último registro
        if ultima_actividad >= INICIO_EXTENSION:
            fin_malla = FIN_PERIODO
        else:
            fin_malla = ultima_actividad

        horas = pd.date_range(
            start=primera_actividad,
            end=fin_malla,
            freq="h",
        )

        bloque = grupo.set_index("fecha_hora")[["salidas", "llegadas"]].reindex(
            horas, fill_value=0
        )

        bloque.index.name = "fecha_hora"
        bloque["estacion_id"] = estacion

        bloques.append(bloque.reset_index())

    resultado = pd.concat(bloques, ignore_index=True)

    resultado = (
        resultado[["estacion_id", "fecha_hora", "salidas", "llegadas"]]
        .sort_values(["estacion_id", "fecha_hora"])
        .reset_index(drop=True)
    )

    # Variables temporales para el modelado
    resultado["hora"] = resultado["fecha_hora"].dt.hour
    resultado["dia_semana"] = resultado["fecha_hora"].dt.dayofweek
    resultado["mes"] = resultado["fecha_hora"].dt.month

    # 0 = lunes, 6 = domingo
    resultado["fin_semana"] = (resultado["dia_semana"] >= 5).astype(int)

    # Variables históricas según la configuración
    for lag in rezagos:
        resultado[f"salidas_lag_{lag}"] = resultado.groupby("estacion_id")["salidas"].shift(lag)

        resultado[f"llegadas_lag_{lag}"] = resultado.groupby("estacion_id")["llegadas"].shift(lag)

    # Días de descanso obligatorio en México
    # Dentro del periodo octubre 2025 - septiembre 2026
    dias_festivos = pd.to_datetime(
        [
            "2025-11-17",
            "2025-12-25",
            "2026-01-01",
            "2026-02-02",
            "2026-03-16",
            "2026-05-01",
            "2026-09-16",
        ]
    )

    # Identificar si la fecha corresponde a un festivo
    resultado["festivo"] = resultado["fecha_hora"].dt.normalize().isin(dias_festivos).astype(int)

    # Codificación cíclica de variables temporales
    variables_ciclicas = {
        "hora": 24,
        "dia_semana": 7,
        "mes": 12,
    }

    for variable, periodo in variables_ciclicas.items():
        inicio = 1 if variable == "mes" else 0

        mapa_sin = {
            valor: sin(2 * pi * (valor - inicio) / periodo)
            for valor in range(inicio, inicio + periodo)
        }

        mapa_cos = {
            valor: cos(2 * pi * (valor - inicio) / periodo)
            for valor in range(inicio, inicio + periodo)
        }

        resultado[f"{variable}_sin"] = resultado[variable].map(mapa_sin)
        resultado[f"{variable}_cos"] = resultado[variable].map(mapa_cos)

    return resultado


def generar_dataset_v3(
    viajes: pd.DataFrame,
    horizonte_horas: int = 1,
    rezagos=None,
):
    """Genera variables para predecir salidas de la siguiente hora."""

    if horizonte_horas != 1:
        raise ValueError("El dataset v3 utiliza un horizonte de 1 hora.")

    # Utilizar los mismos rezagos del Pipeline
    if rezagos is None:
        rezagos = cargar_parametros()["features"]["rezagos"]

    resultado = generar_dataset_estacion_hora(viajes, rezagos=rezagos)

    por_estacion = resultado.groupby("estacion_id", sort=False)

    # La fecha_hora representa la hora actual (t)
    # El objetivo corresponde a las salidas de la siguiente hora (t+1)
    resultado["fecha_objetivo"] = por_estacion["fecha_hora"].shift(-1)
    resultado["salidas_siguiente_hora"] = por_estacion["salidas"].shift(-1)

    # Alinear las variables temporales con la hora que se predice
    columnas_temporales = [
        "hora",
        "dia_semana",
        "mes",
        "fin_semana",
        "festivo",
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
    ]

    resultado[columnas_temporales] = por_estacion[columnas_temporales].shift(-1)

    # Alinear los rezagos con la hora objetivo
    # lag_1 representa la demanda de la hora actual
    columnas_lags = [f"{tipo}_lag_{lag}" for tipo in ["salidas", "llegadas"] for lag in rezagos]

    resultado[columnas_lags] = por_estacion[columnas_lags].shift(-1)

    # Conservar únicamente los registros con objetivo conocido
    siguiente_hora = resultado["fecha_hora"] + pd.Timedelta(hours=1)

    resultado = resultado[resultado["fecha_objetivo"] == siguiente_hora].copy()

    resultado["salidas_siguiente_hora"] = resultado["salidas_siguiente_hora"].astype("int32")

    # Los conteos actuales ya están representados por lag_1
    resultado = resultado.drop(columns=["salidas", "llegadas"])

    return resultado.reset_index(drop=True)


def normalizar_variables_historicas(datos: pd.DataFrame, rezagos=None):
    """Normaliza los lags utilizando solo los datos de entrenamiento."""

    if rezagos is None:
        rezagos = cargar_parametros()["features"]["rezagos"]

    columnas = [f"{tipo}_lag_{lag}" for tipo in ["salidas", "llegadas"] for lag in rezagos]

    resultado = datos.copy()

    # Entrenamiento: fechas anteriores a agosto de 2026
    entrenamiento = resultado["fecha_hora"] < FECHA_CORTE

    if not entrenamiento.any():
        raise ValueError("No hay registros para entrenar el escalador.")

    # Identificar registros sin historial completo
    resultado["historial_incompleto"] = resultado[columnas].isna().any(axis=1).astype("int8")

    # Calcular medianas solamente con entrenamiento
    medianas = resultado.loc[entrenamiento, columnas].median()

    if medianas.isna().any():
        raise ValueError("No se pudieron calcular todas las medianas.")

    # Imputar datos históricos faltantes sin usar datos futuros
    valores = resultado[columnas].fillna(medianas)

    # Ajustar el escalador exclusivamente con entrenamiento
    escalador = StandardScaler()
    escalador.fit(valores.loc[entrenamiento])

    # Transformar todos los registros con el mismo escalador
    columnas_normalizadas = [f"{c}_norm" for c in columnas]

    resultado[columnas_normalizadas] = escalador.transform(valores).astype("float32")

    return resultado, escalador, medianas.to_dict()


def crear_codificador_estaciones(datos: pd.DataFrame):
    """Prepara la codificación de estaciones con datos de entrenamiento."""

    entrenamiento = datos.loc[
        datos["fecha_hora"] < FECHA_CORTE,
        ["estacion_id"],
    ]

    if entrenamiento.empty:
        raise ValueError("No hay estaciones en entrenamiento.")

    # Las estaciones no tienen un orden numérico significativo
    codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # Aprender únicamente las estaciones del entrenamiento
    codificador.fit(entrenamiento)

    logger.info(f"Estaciones identificadas: {len(codificador.categories_[0])}")

    return codificador


@app.command("v2")
def main(
    input_path: Path = INTERIM_DATA_DIR / "viajes_limpios.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "estacion_hora.parquet",
):
    """Lee los viajes limpios y genera el dataset estación-hora."""

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset limpio: {input_path}")

    logger.info("Cargando viajes limpios...")

    columnas = [
        "Origen_Id",
        "Destino_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
    ]

    viajes = pd.read_parquet(input_path, columns=columnas)

    logger.info(f"Viajes cargados: {len(viajes):,}")

    rezagos = cargar_parametros()["features"]["rezagos"]

    procesados = generar_dataset_estacion_hora(viajes, rezagos=rezagos)

    # Normalizar las variables históricas
    procesados, escalador, medianas = normalizar_variables_historicas(procesados, rezagos=rezagos)

    ruta_escalador = PROCESSED_DATA_DIR / "escalador_lags.pkl"
    ruta_escalador.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_escalador, "wb") as archivo:
        pickle.dump(
            {
                "escalador": escalador,
                "medianas": medianas,
                "fecha_corte": str(FECHA_CORTE.date()),
            },
            archivo,
        )

    logger.info(f"Escalador guardado en: {ruta_escalador}")

    # Preparar codificación de estaciones
    codificador = crear_codificador_estaciones(procesados)

    # Guardar codificador para reutilizarlo en el modelado
    ruta_codificador = PROCESSED_DATA_DIR / "codificador_estaciones.pkl"

    with open(ruta_codificador, "wb") as archivo:
        pickle.dump(codificador, archivo)

    logger.info(f"Codificador guardado en: {ruta_codificador}")

    # Contabilizar llegadas dentro del periodo
    llegadas_validas = int((viajes["Fin_del_viaje"].dt.floor("h") <= FIN_PERIODO).sum())

    llegadas_fuera_periodo = len(viajes) - llegadas_validas

    # Validar resultados antes de guardar el dataset
    assert procesados["salidas"].sum() == len(viajes)
    assert procesados["llegadas"].sum() == llegadas_validas
    assert procesados["fecha_hora"].max() <= FIN_PERIODO

    logger.info(f"Llegadas fuera del periodo: {llegadas_fuera_periodo:,}")

    # Guardar los datos después de validar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    procesados.to_parquet(output_path, index=False)

    logger.success(f"Dataset generado: {len(procesados):,} registros")

    logger.info(f"Total de salidas: {procesados['salidas'].sum():,}")
    logger.info(f"Total de llegadas: {procesados['llegadas'].sum():,}")
    logger.info(f"Archivo guardado en: {output_path}")

    # Configurar MLflow utilizando el experimento del equipo
    configurar_mlflow()

    # Obtener información del código que generó el dataset
    commit, cambios_sin_commit = estado_git()

    if cambios_sin_commit:
        logger.warning(
            "Existen cambios sin commit. "
            "El hash de Git no representa exactamente el código actual."
        )

    # Registrar el dataset procesado en MLflow
    with mlflow.start_run(run_name="preprocesamiento_v2"):
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_cambios_sin_commit": str(cambios_sin_commit).lower(),
                "fuente_datos": URL_DATOS_ABIERTOS,
            }
        )

        # Registrar el dataset limpio como entrada del preprocesamiento
        viajes_limpios = pd.read_parquet(input_path)

        dataset_interim = mlflow.data.from_pandas(
            viajes_limpios,
            source=str(input_path),
            name="mibici_interim_v1",
        )

        mlflow.log_input(dataset_interim, context="interim")

        # Liberar memoria después del registro
        del viajes_limpios

        # Registrar la versión del dataset
        dataset = mlflow.data.from_pandas(
            procesados,
            source=str(output_path),
            name="mibici_processed_v2",
        )

        mlflow.log_input(dataset, context="processed")

        # Registrar parámetros del preprocesamiento
        mlflow.log_params(
            {
                "filas_procesadas": len(procesados),
                "estaciones_entrenamiento": len(codificador.categories_[0]),
                "lags": ",".join(map(str, rezagos)),
                "fecha_corte_entrenamiento": str(FECHA_CORTE.date()),
            }
        )

        # Conservar dataset y transformadores
        mlflow.log_artifact(
            str(output_path),
            artifact_path="datos",
        )

        mlflow.log_artifact(
            str(ruta_escalador),
            artifact_path="transformadores",
        )

        mlflow.log_artifact(
            str(ruta_codificador),
            artifact_path="transformadores",
        )

    logger.success("Dataset mibici_processed_v2 registrado en MLflow.")


@app.command("v3")
def generar_v3(
    input_path: Path = INTERIM_DATA_DIR / "viajes_limpios.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
):
    """Genera y registra el dataset v3 para modelado."""

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el dataset limpio: {input_path}")

    params = cargar_parametros()
    horizonte = int(params["features"]["horizonte_horas"])
    rezagos = params["features"]["rezagos"]

    columnas = [
        "Origen_Id",
        "Destino_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
    ]

    logger.info("Cargando viajes limpios...")
    viajes = pd.read_parquet(input_path, columns=columnas)

    logger.info("Generando dataset v3 sin escalar...")
    procesados = generar_dataset_v3(
        viajes,
        horizonte_horas=horizonte,
        rezagos=rezagos,
    )

    # Verificar que el objetivo corresponde a la siguiente hora
    fecha_esperada = procesados["fecha_hora"] + pd.Timedelta(hours=horizonte)

    if not (procesados["fecha_objetivo"] == fecha_esperada).all():
        raise ValueError("El objetivo no está alineado con el horizonte.")

    if procesados["salidas_siguiente_hora"].isna().any():
        raise ValueError("Existen objetivos sin valor conocido.")

    if procesados["fecha_objetivo"].max() > FIN_PERIODO:
        raise ValueError("Existen objetivos fuera del periodo.")

    # El escalado y la codificación se harán dentro del Pipeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    procesados.to_parquet(output_path, index=False)

    logger.success(f"Dataset v3 generado: {len(procesados):,} registros")
    logger.info(f"Archivo guardado en: {output_path}")

    # Utilizar el experimento de la Fase 2
    configurar_mlflow_modelado(params["mlflow"]["experimento"])

    commit, cambios_sin_commit = estado_git()

    if cambios_sin_commit:
        logger.warning("Existen cambios sin commit en el repositorio.")

    with mlflow.start_run(run_name="preprocesamiento_v3"):
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_cambios_sin_commit": str(cambios_sin_commit).lower(),
                "fuente_datos": URL_DATOS_ABIERTOS,
            }
        )

        # Registrar el dataset limpio como entrada
        viajes_completos = pd.read_parquet(input_path)

        dataset_interim = mlflow.data.from_pandas(
            viajes_completos,
            source=str(input_path),
            name="mibici_interim_v1",
        )

        mlflow.log_input(dataset_interim, context="interim")
        del viajes_completos

        # Registrar la nueva versión procesada
        dataset_v3 = mlflow.data.from_pandas(
            procesados,
            source=str(output_path),
            name="mibici_processed_v3",
        )

        mlflow.log_input(dataset_v3, context="processed")

        mlflow.log_params(
            {
                "filas_procesadas": len(procesados),
                "estaciones": procesados["estacion_id"].nunique(),
                "horizonte_horas": horizonte,
                "rezagos": ",".join(map(str, params["features"]["rezagos"])),
                "escalado_previo": False,
            }
        )

        mlflow.log_artifact(str(output_path), artifact_path="datos")

    logger.success("Dataset mibici_processed_v3 registrado en MLflow.")


if __name__ == "__main__":
    app()
