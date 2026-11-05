"""Inferencia y generación de predicciones utilizando el modelo champion de MLflow."""

from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR, cargar_parametros
from mibici.modeling.tracking import configurar_mlflow

app = typer.Typer()


def cargar_modelo(
    nombre: str | None = None,
    alias: str = "champion",
    experimento: str | None = None,
):
    """Carga un modelo registrado desde el MLflow Model Registry utilizando su alias."""
    params = cargar_parametros()
    experimento = experimento or params["mlflow"]["experimento"]
    configurar_mlflow(experimento)

    nombre = nombre or params["mlflow"]["modelo_registrado"]
    uri = f"models:/{nombre}@{alias}"

    logger.info(f"Cargando modelo desde MLflow Model Registry: {uri}")
    return mlflow.sklearn.load_model(uri)


def predecir(
    datos: pd.DataFrame,
    modelo=None,
    nombre: str | None = None,
    alias: str = "champion",
) -> np.ndarray:
    """Genera predicciones sobre un conjunto de datos descartando la variable objetivo si existe."""
    if modelo is None:
        modelo = cargar_modelo(nombre=nombre, alias=alias)

    X = datos.drop(columns=["salidas_siguiente_hora"], errors="ignore")
    return modelo.predict(X)


@app.command()
def main(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
    salida_path: Path = PROCESSED_DATA_DIR / "predicciones.parquet",
    alias: str = "champion",
    modelo_nombre: str | None = None,
):
    """Ejecuta inferencia por lotes sobre un dataset y guarda las predicciones."""
    if not datos_path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {datos_path}")

    logger.info(f"Cargando datos para inferencia: {datos_path}")
    datos = (
        pd.read_parquet(datos_path) if datos_path.suffix == ".parquet" else pd.read_csv(datos_path)
    )

    modelo = cargar_modelo(nombre=modelo_nombre, alias=alias)
    y_pred = predecir(datos, modelo=modelo)

    columnas_id = [
        col for col in ["estacion_id", "fecha_hora", "fecha_objetivo"] if col in datos.columns
    ]
    salida = datos[columnas_id].copy() if columnas_id else pd.DataFrame(index=datos.index)

    if "salidas_siguiente_hora" in datos.columns:
        salida["salidas_reales"] = datos["salidas_siguiente_hora"]

    salida["salidas_predichas"] = y_pred

    salida_path.parent.mkdir(parents=True, exist_ok=True)
    if salida_path.suffix == ".parquet":
        salida.to_parquet(salida_path, index=False)
    else:
        salida.to_csv(salida_path, index=False)

    logger.success(f"{len(salida):,} predicciones guardadas en: {salida_path}")


if __name__ == "__main__":
    app()
