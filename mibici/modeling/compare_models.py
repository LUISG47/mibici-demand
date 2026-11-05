
"""Comparación y registro de modelos de demanda en MLflow."""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import PROCESSED_DATA_DIR, cargar_parametros
from mibici.modeling.evaluate_baselines import evaluar_baselines
from mibici.modeling.tracking import configurar_mlflow, registrar_evaluacion
from mibici.modeling.tuning import (
    evaluar_hgb_ajustado,
    evaluar_ridge_ajustado,
)

app = typer.Typer()


def comparar_modelos(
    datos: pd.DataFrame,
    params=None,
    checkpoint_dir: Path = PROCESSED_DATA_DIR / "checkpoints_hgb",
):
    """Evalúa los cuatro modelos y registra sus resultados en MLflow."""
    if params is None:
        params = cargar_parametros()

    resultados = evaluar_baselines(datos, params=params)

    logger.info("Evaluando Ridge ajustado...")
    tabla_ridge, resumen_ridge = evaluar_ridge_ajustado(
        datos,
        params=params,
    )
    resultados["ridge_ajustado"] = {
        "tabla": tabla_ridge,
        "resumen": resumen_ridge,
    }

    logger.info("Recuperando evaluación de HGB Poisson...")
    tabla_hgb, resumen_hgb = evaluar_hgb_ajustado(
        datos,
        params=params,
        checkpoint_dir=checkpoint_dir,
    )
    resultados["hgb_poisson"] = {
        "tabla": tabla_hgb,
        "resumen": resumen_hgb,
    }

    configurar_mlflow(params["mlflow"]["experimento"])

    for nombre, resultado in resultados.items():
        hiperparametros = {
            "algoritmo": nombre,
            "horizonte_horas": params["features"]["horizonte_horas"],
            "rezagos": ",".join(map(str, params["features"]["rezagos"])),
            "validacion": "walk_forward",
        }

        run_id = registrar_evaluacion(
            nombre_modelo=nombre,
            tabla=resultado["tabla"],
            resumen=resultado["resumen"],
            hiperparametros=hiperparametros,
        )

        logger.success(f"Modelo {nombre} registrado. Run ID: {run_id}")

    comparacion = pd.DataFrame(
        [
            {
                "modelo": nombre,
                **resultado["resumen"],
            }
            for nombre, resultado in resultados.items()
        ]
    )

    comparacion = comparacion.sort_values(
        "mae_media"
    ).reset_index(drop=True)

    return comparacion


@app.command()
def main(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
):
    """Compara los cuatro modelos y registra las evaluaciones en MLflow."""
    if not datos_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {datos_path}"
        )

    logger.info(f"Cargando dataset: {datos_path}")
    datos = pd.read_parquet(datos_path)

    comparacion = comparar_modelos(datos)

    logger.success("Comparación registrada en MLflow.")
    logger.info(f"\n{comparacion.to_string(index=False)}")


if __name__ == "__main__":
    app()
