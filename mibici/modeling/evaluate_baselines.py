"""Evaluación walk-forward de las líneas base de MiBici."""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR, cargar_parametros
from mibici.modeling.baselines import (
    HistoricalMeanBaseline,
    WeeklyLagBaseline,
)
from mibici.modeling.validation import (
    WalkForwardSplitter,
    evaluar_walk_forward,
)

app = typer.Typer()


def evaluar_baselines(datos: pd.DataFrame, params=None):
    """Evalúa ambas líneas base sobre las mismas ventanas temporales."""

    if params is None:
        params = cargar_parametros()

    columnas = [
        "estacion_id",
        "fecha_hora",
        "dia_semana",
        "hora",
        "salidas_lag_168",
        "salidas_siguiente_hora",
    ]

    faltantes = set(columnas) - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    modelos = {
        "baseline_lag_168": WeeklyLagBaseline(),
        "baseline_media_historica": HistoricalMeanBaseline(),
    }

    splitter = WalkForwardSplitter.desde_parametros(params)

    resultados = {}

    for nombre, modelo in modelos.items():
        logger.info(f"Evaluando {nombre}...")

        tabla, resumen = evaluar_walk_forward(
            modelo=modelo,
            X=X,
            y=y,
            splitter=splitter,
        )

        resultados[nombre] = {
            "tabla": tabla,
            "resumen": resumen,
        }

        logger.info(f"Resultados de {nombre}:")
        logger.info(f"\n{tabla.to_string(index=False)}")
        logger.info(f"Métricas promedio: {resumen}")

    return resultados


@app.command()
def main(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
):
    """Ejecuta ambas líneas base utilizando el dataset v3."""

    if not datos_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {datos_path}")

    logger.info(f"Cargando dataset: {datos_path}")

    datos = pd.read_parquet(datos_path)

    resultados = evaluar_baselines(datos)

    comparacion = pd.DataFrame(
        [
            {
                "modelo": nombre,
                **resultado["resumen"],
            }
            for nombre, resultado in resultados.items()
        ]
    )

    logger.success("Evaluación de líneas base completada.")
    logger.info(f"\n{comparacion.to_string(index=False)}")


if __name__ == "__main__":
    app()
