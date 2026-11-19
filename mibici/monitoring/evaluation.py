"""Evaluación del impacto de data drift sobre el desempeño del modelo."""

import numpy as np
import pandas as pd

from mibici.modeling.predict import predecir
from mibici.modeling.validation import METRICAS
from mibici.monitoring.drift import ESCENARIOS, simular_drift

COLUMNA_OBJETIVO = "salidas_siguiente_hora"


def calcular_metricas(
    datos: pd.DataFrame,
    modelo,
) -> dict[str, float]:
    """Calcula las métricas del modelo sobre un conjunto etiquetado."""
    if COLUMNA_OBJETIVO not in datos.columns:
        raise ValueError(f"Falta la variable objetivo: {COLUMNA_OBJETIVO}")

    if datos.empty:
        raise ValueError("No se puede evaluar un conjunto de datos vacío.")

    reales = datos[COLUMNA_OBJETIVO].to_numpy(dtype=float)

    if not np.isfinite(reales).all():
        raise ValueError("La variable objetivo contiene valores no finitos.")

    predicciones = predecir(datos, modelo=modelo)

    if not np.isfinite(predicciones).all():
        raise ValueError("El modelo generó predicciones no finitas.")

    return {nombre: funcion(reales, predicciones) for nombre, funcion in METRICAS.items()}


def evaluar_escenarios(
    datos: pd.DataFrame,
    modelo,
    params: dict | None = None,
) -> pd.DataFrame:
    """Compara el desempeño original contra los escenarios sintéticos de drift."""
    referencia = calcular_metricas(datos, modelo)
    filas = [
        {
            "escenario": "referencia",
            "n_registros": len(datos),
            **referencia,
            "deterioro_mae_relativo": 0.0,
        }
    ]

    for escenario in ESCENARIOS:
        alterados = simular_drift(datos, escenario, params=params)
        metricas = calcular_metricas(alterados, modelo)

        mae_base = referencia["mae"]
        if np.isclose(mae_base, 0):
            deterioro = 0.0 if np.isclose(metricas["mae"], 0) else float("inf")
        else:
            deterioro = (metricas["mae"] - mae_base) / mae_base

        filas.append(
            {
                "escenario": escenario,
                "n_registros": len(datos),
                **metricas,
                "deterioro_mae_relativo": float(deterioro),
            }
        )

    return pd.DataFrame(filas)
