"""Entrenamiento y evaluación de modelos para MiBici."""

from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
import typer

from mibici.config import PROCESSED_DATA_DIR, cargar_parametros
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.tracking import configurar_mlflow, registrar_evaluacion
from mibici.modeling.validation import WalkForwardSplitter, evaluar_walk_forward

app = typer.Typer()


def entrenar_y_evaluar(
    datos: pd.DataFrame,
    estimador,
    params=None,
    registrar=False,
    nombre_modelo="ridge_inicial",
):
    """Entrena y evalúa un estimador con validación walk-forward."""

    if params is None:
        params = cargar_parametros()

    if "fecha_hora" not in datos.columns:
        raise ValueError("Falta fecha_hora para la validación temporal.")

    if "salidas_siguiente_hora" not in datos.columns:
        raise ValueError("Falta la variable objetivo.")

    # Separar predictores y objetivo
    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    # El Pipeline descarta las fechas antes del estimador
    modelo = crear_pipeline(estimador, params=params)

    # Crear las ventanas definidas en params.yaml
    splitter = WalkForwardSplitter.desde_parametros(params)

    # En cada ventana se ajusta nuevamente todo el Pipeline
    tabla, resumen = evaluar_walk_forward(modelo, X, y, splitter)

    logger.info("Resultados por ventana:")
    logger.info(f"\n{tabla.to_string(index=False)}")
    logger.info(f"Métricas promedio: {resumen}")

    # Ajustar un modelo final con todos los datos disponibles
    modelo_final = clone(modelo).fit(X, y)

    if registrar:
        configurar_mlflow(params["mlflow"]["experimento"])

        hiperparametros = {
            "algoritmo": type(estimador).__name__,
            "horizonte_horas": params["features"]["horizonte_horas"],
            "rezagos": ",".join(map(str, params["features"]["rezagos"])),
        }

        for clave, valor in estimador.get_params(deep=False).items():
            hiperparametros[f"modelo_{clave}"] = str(valor)

        registrar_evaluacion(
            nombre_modelo=nombre_modelo,
            tabla=tabla,
            resumen=resumen,
            hiperparametros=hiperparametros,
            modelo_final=modelo_final,
            X_ejemplo=X.head(5),
        )

    return modelo_final, tabla, resumen


@app.command()
def main(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
    registrar: bool = False,
):
    """Entrena un modelo Ridge inicial utilizando el dataset v3."""

    if not datos_path.exists():
        raise FileNotFoundError(f"No existe el dataset v3: {datos_path}")

    params = cargar_parametros()

    logger.info(f"Cargando dataset: {datos_path}")
    datos = pd.read_parquet(datos_path)

    # Valor inicial del espacio de hiperparámetros definido en YAML
    alpha = params["modelado"]["hiperparametros"]["ridge"]["alpha"][0]
    estimador = Ridge(alpha=alpha)

    _, tabla, resumen = entrenar_y_evaluar(
        datos=datos,
        estimador=estimador,
        params=params,
        registrar=registrar,
        nombre_modelo="ridge_inicial",
    )

    logger.success(f"Evaluación completada: {len(tabla)} ventanas")
    logger.info(f"MAE promedio: {resumen['mae_media']:.4f}")
    logger.info(f"RMSE promedio: {resumen['rmse_media']:.4f}")
    logger.info(f"WAPE promedio: {resumen['wape_media']:.4f}")


if __name__ == "__main__":
    app()
