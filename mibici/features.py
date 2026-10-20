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
from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, configurar_mlflow
from mibici.dataset import URL_DATOS_ABIERTOS

app = typer.Typer()


def contar_eventos(estaciones, fechas, nombre):
    """Cuenta eventos por estación y hora."""

    eventos = pd.DataFrame(
        {
            "estacion_id": estaciones,
            "fecha_hora": fechas.dt.floor("h"),
        }
    )

    return eventos.groupby(["estacion_id", "fecha_hora"]).size().rename(nombre)


def generar_dataset_estacion_hora(viajes):
    """Genera los conteos horarios de salidas y llegadas."""

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

    # Incorporar horas sin viajes para cada estación
    bloques = []

    for estacion, grupo in conteos.groupby("estacion_id"):
        horas = pd.date_range(
            start=grupo["fecha_hora"].min(),
            end=grupo["fecha_hora"].max(),
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

    # Variables históricas de demanda por estación
    for lag in [1, 24, 168]:
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


def normalizar_variables_historicas(datos: pd.DataFrame):
    """Normaliza los lags utilizando solo los datos de entrenamiento."""

    columnas = [f"{tipo}_lag_{lag}" for tipo in ["salidas", "llegadas"] for lag in [1, 24, 168]]

    resultado = datos.copy()

    # Entrenamiento: fechas anteriores a agosto de 2026
    fecha_corte = pd.Timestamp("2026-08-01")
    entrenamiento = resultado["fecha_hora"] < fecha_corte

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

    fecha_corte = pd.Timestamp("2026-08-01")

    entrenamiento = datos.loc[datos["fecha_hora"] < fecha_corte, ["estacion_id"]]

    if entrenamiento.empty:
        raise ValueError("No hay estaciones en entrenamiento.")

    # Las estaciones no tienen un orden numérico significativo
    codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # Aprender únicamente las estaciones del entrenamiento
    codificador.fit(entrenamiento)

    logger.info(f"Estaciones identificadas: {len(codificador.categories_[0])}")

    return codificador


@app.command()
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

    procesados = generar_dataset_estacion_hora(viajes)

    # Normalizar las variables históricas
    procesados, escalador, medianas = normalizar_variables_historicas(procesados)

    ruta_escalador = PROCESSED_DATA_DIR / "escalador_lags.pkl"
    ruta_escalador.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_escalador, "wb") as archivo:
        pickle.dump(
            {
                "escalador": escalador,
                "medianas": medianas,
                "fecha_corte": "2026-08-01",
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    procesados.to_parquet(output_path, index=False)

    # Verificar que no se perdieron eventos durante la agregación
    assert procesados["salidas"].sum() == len(viajes)
    assert procesados["llegadas"].sum() == len(viajes)

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
    with mlflow.start_run(run_name="preprocesamiento"):
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_cambios_sin_commit": str(cambios_sin_commit).lower(),
                "fuente_datos": URL_DATOS_ABIERTOS,
            }
        )

        # Registrar la versión del dataset
        dataset = mlflow.data.from_pandas(
            procesados,
            source=str(output_path),
            name="mibici_processed_v1",
        )

        mlflow.log_input(dataset, context="processed")

        # Registrar parámetros del preprocesamiento
        mlflow.log_params(
            {
                "filas_procesadas": len(procesados),
                "estaciones_entrenamiento": len(codificador.categories_[0]),
                "lags": "1,24,168",
                "fecha_corte_entrenamiento": "2026-08-01",
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

    logger.success("Dataset mibici_processed_v1 registrado en MLflow.")


if __name__ == "__main__":
    app()
