"""Comparación y registro de modelos de demanda en MLflow."""

from collections import Counter
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import typer

from mibici.config import PROCESSED_DATA_DIR, cargar_parametros
from mibici.modeling.baselines import HistoricalMeanBaseline, WeeklyLagBaseline
from mibici.modeling.evaluate_baselines import evaluar_baselines
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.tracking import (
    configurar_mlflow,
    registrar_evaluacion,
    registrar_modelo,
)
from mibici.modeling.tuning import (
    evaluar_hgb_ajustado,
    evaluar_ridge_ajustado,
)

app = typer.Typer()

# Nombre del modelo en la comparación -> clave de su espacio en params.yaml
ESPACIOS = {
    "ridge_ajustado": "ridge",
    "hgb_poisson": "hist_gradient_boosting",
}


def hiperparametros_frecuentes(nombre, tabla, params):
    """Regresa la combinación de hiperparámetros elegida en más ventanas.

    Las líneas base no tienen hiperparámetros. Si hay empate, gana la combinación
    de la ventana más reciente (entrenada con más datos).
    """
    if nombre not in ESPACIOS:
        return {}

    espacio = params["modelado"]["hiperparametros"][ESPACIOS[nombre]]
    columnas = list(espacio)

    # Recorrer de la ventana más reciente a la más antigua para desempatar
    combinaciones = [tuple(fila) for fila in tabla[columnas].iloc[::-1].itertuples(index=False)]
    mas_frecuente = Counter(combinaciones).most_common(1)[0][0]

    # Convertir tipos de NumPy a tipos de Python (MLflow y los estimadores)
    return {
        columna: (valor.item() if isinstance(valor, np.generic) else valor)
        for columna, valor in zip(columnas, mas_frecuente, strict=True)
    }


def crear_modelo_final(nombre, hiperparametros, params):
    """Construye (sin entrenar) el modelo indicado con sus hiperparámetros."""
    if nombre == "baseline_lag_168":
        return WeeklyLagBaseline()

    if nombre == "baseline_media_historica":
        return HistoricalMeanBaseline()

    if nombre == "ridge_ajustado":
        return crear_pipeline(Ridge(**hiperparametros), params=params)

    if nombre == "hgb_poisson":
        # max_iter y max_leaf_nodes llegan como float desde la tabla por ventana
        enteros = {"max_iter", "max_leaf_nodes", "min_samples_leaf"}
        hiperparametros = {
            clave: int(valor) if clave in enteros else valor
            for clave, valor in hiperparametros.items()
        }
        estimador = HistGradientBoostingRegressor(
            **hiperparametros,
            random_state=params["modelado"]["semilla"],
        )
        return crear_pipeline(estimador, params=params)

    raise ValueError(f"Modelo desconocido: {nombre}")


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

    # Mejor modelo: menor MAE promedio en las ventanas
    mejor = min(resultados, key=lambda nombre: resultados[nombre]["resumen"]["mae_media"])

    # Entrenar el mejor modelo con todo el periodo
    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    hiperparametros_mejor = hiperparametros_frecuentes(mejor, resultados[mejor]["tabla"], params)

    logger.info(f"Entrenando modelo final ({mejor}) con todo el periodo...")
    modelo_final = crear_modelo_final(mejor, hiperparametros_mejor, params).fit(X, y)

    configurar_mlflow(params["mlflow"]["experimento"])

    for nombre, resultado in resultados.items():
        # Los hiperparámetros por ventana quedan en tabla_ventanas.csv (artefacto);
        # como params se registra la combinación más frecuente.
        elegidos = hiperparametros_frecuentes(nombre, resultado["tabla"], params)

        hiperparametros = {
            "algoritmo": nombre,
            "horizonte_horas": params["features"]["horizonte_horas"],
            "rezagos": ",".join(map(str, params["features"]["rezagos"])),
            "validacion": "walk_forward",
            **elegidos,
        }

        es_mejor = nombre == mejor

        run_id = registrar_evaluacion(
            nombre_modelo=nombre,
            tabla=resultado["tabla"],
            resumen=resultado["resumen"],
            hiperparametros=hiperparametros,
            modelo_final=modelo_final if es_mejor else None,
            X_ejemplo=X if es_mejor else None,
        )

        logger.success(f"Modelo {nombre} registrado. Run ID: {run_id}")

        if es_mejor:
            version = registrar_modelo(
                run_id,
                hiperparametros=hiperparametros,
                metricas=resultado["resumen"],
                alias="champion",
            )
            logger.success(
                f"{nombre} registrado en el Model Registry: "
                f"versión {version.version} con alias champion."
            )

    comparacion = pd.DataFrame(
        [
            {
                "modelo": nombre,
                **resultado["resumen"],
            }
            for nombre, resultado in resultados.items()
        ]
    )

    comparacion = comparacion.sort_values("mae_media").reset_index(drop=True)

    return comparacion


@app.command()
def main(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
):
    """Compara los cuatro modelos y registra las evaluaciones en MLflow."""
    if not datos_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {datos_path}")

    logger.info(f"Cargando dataset: {datos_path}")
    datos = pd.read_parquet(datos_path)

    comparacion = comparar_modelos(datos)

    logger.success("Comparación registrada en MLflow.")
    logger.info(f"\n{comparacion.to_string(index=False)}")


if __name__ == "__main__":
    app()
