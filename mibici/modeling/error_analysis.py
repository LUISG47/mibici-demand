"""Diagnóstico de errores de predicción de HGB Poisson."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici.config import cargar_parametros
from mibici.modeling.checkpoints import cargar_checkpoint
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.validation import METRICAS, WalkForwardSplitter


def obtener_predicciones_hgb(
    datos: pd.DataFrame,
    ruta_checkpoint: Path,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconstruye una ventana de HGB y comprueba sus métricas originales."""

    if params is None:
        params = cargar_parametros()

    checkpoint = cargar_checkpoint(ruta_checkpoint)

    if checkpoint is None:
        raise FileNotFoundError(f"No existe el checkpoint: {ruta_checkpoint}")

    resultado = checkpoint.get("resultado")

    if not isinstance(resultado, dict):
        raise TypeError("El checkpoint no contiene un resultado válido.")

    requeridas = {
        "fecha_hora",
        "estacion_id",
        "hora",
        "salidas_siguiente_hora",
    }

    faltantes = requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    # Los hiperparámetros a reconstruir son los del espacio de búsqueda de params.yaml
    espacio = params["modelado"]["hiperparametros"]["hist_gradient_boosting"]
    hiperparametros = {nombre: resultado[nombre] for nombre in espacio}

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    splitter = WalkForwardSplitter.desde_parametros(params)
    mes_checkpoint = resultado["mes_prueba"]

    ventanas = [
        (entrenamiento, prueba)
        for indice, (entrenamiento, prueba) in enumerate(splitter.split(X, y))
        if str(splitter.meses_prueba[indice]) == mes_checkpoint
    ]

    if len(ventanas) != 1:
        raise ValueError(f"No se encontró una ventana única para {mes_checkpoint}.")

    entrenamiento, prueba = ventanas[0]

    if len(entrenamiento) != resultado["n_entrenamiento"] or len(prueba) != resultado["n_prueba"]:
        raise ValueError("Los tamaños de la ventana no coinciden con el checkpoint.")

    estimador = HistGradientBoostingRegressor(
        **hiperparametros,
        random_state=params["modelado"]["semilla"],
    )

    modelo = crear_pipeline(estimador, params=params)

    modelo.fit(
        X.iloc[entrenamiento],
        y[entrenamiento],
    )

    predicciones = modelo.predict(X.iloc[prueba])
    valores_reales = y[prueba]

    if not np.isfinite(predicciones).all():
        raise ValueError("El modelo generó predicciones no finitas.")

    metricas = {
        nombre: metrica(valores_reales, predicciones) for nombre, metrica in METRICAS.items()
    }

    for nombre, valor in metricas.items():
        if not np.isclose(
            valor,
            resultado[nombre],
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                f"La métrica {nombre} no coincide: "
                f"calculada={valor}, "
                f"checkpoint={resultado[nombre]}"
            )

    diagnostico = X.iloc[prueba][["fecha_hora", "estacion_id", "hora"]].copy()

    diagnostico["valor_real"] = valores_reales
    diagnostico["prediccion"] = predicciones

    diagnostico["error"] = diagnostico["valor_real"] - diagnostico["prediccion"]

    diagnostico["error_absoluto"] = diagnostico["error"].abs()

    diagnostico["error_cuadratico"] = diagnostico["error"] ** 2

    return diagnostico, metricas
