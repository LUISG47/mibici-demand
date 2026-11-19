
"""Clasificación de alertas de data drift y acciones recomendadas."""

import numpy as np
import pandas as pd

from mibici.config import cargar_parametros


def evaluar_alertas(
    deteccion: pd.DataFrame,
    deterioro_mae_relativo: float,
    params: dict | None = None,
) -> dict:
    """Clasifica drift estadístico y deterioro de desempeño.

    El deterioro se mide respecto al MAE calculado sobre los mismos
    registros sin alteraciones, no respecto al MAE histórico del Issue.
    """
    if params is None:
        params = cargar_parametros()

    umbrales = params["drift"]["umbrales"]

    psi_advertencia = float(umbrales["psi_advertencia"])
    psi_critico = float(umbrales["psi_critico"])
    ks_pvalor = float(umbrales["ks_pvalor"])
    deterioro_limite = float(umbrales["deterioro_mae_relativo"])

    if not (
        0 <= psi_advertencia < psi_critico
        and 0 < ks_pvalor < 1
        and deterioro_limite >= 0
    ):
        raise ValueError("Los umbrales de drift no son válidos.")

    if deteccion.empty:
        raise ValueError("La tabla de detección no puede estar vacía.")

    requeridas = {
        "variable",
        "psi",
        "ks_pvalor",
        "nulos_referencia",
        "nulos_actual",
    }
    faltantes = requeridas - set(deteccion.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas de detección: {sorted(faltantes)}")

    if deteccion["variable"].duplicated().any():
        raise ValueError("La tabla contiene variables duplicadas.")

    if deteccion[list(requeridas - {"variable"})].isna().any().any():
        raise ValueError("La detección contiene métricas nulas.")

    if not np.isfinite(deterioro_mae_relativo):
        raise ValueError("El deterioro del MAE debe ser finito.")

    tabla = deteccion.copy(deep=True)

    # Los nulos se analizan explícitamente además del PSI y KS.
    tabla["aumento_nulos"] = (
        tabla["nulos_actual"] - tabla["nulos_referencia"]
    )

    tabla["alerta_psi"] = tabla["psi"] >= psi_advertencia
    tabla["alerta_ks"] = tabla["ks_pvalor"] < ks_pvalor
    tabla["alerta_nulos"] = tabla["aumento_nulos"] > 0

    tabla["drift_detectado"] = (
        tabla["alerta_psi"]
        | tabla["alerta_ks"]
        | tabla["alerta_nulos"]
    )

    tabla["nivel"] = "sin_alerta"
    tabla.loc[tabla["drift_detectado"], "nivel"] = "advertencia"
    tabla.loc[tabla["psi"] >= psi_critico, "nivel"] = "critico"

    hay_drift = bool(tabla["drift_detectado"].any())
    hay_drift_critico = bool((tabla["nivel"] == "critico").any())
    hay_deterioro = bool(deterioro_mae_relativo >= deterioro_limite)

    if hay_drift_critico or (hay_drift and hay_deterioro):
        nivel_general = "critico"
    elif hay_drift or hay_deterioro:
        nivel_general = "advertencia"
    else:
        nivel_general = "sin_alerta"

    acciones = []

    if hay_drift:
        acciones.append("Revisar el pipeline de características y la calidad de datos.")

    if hay_deterioro:
        acciones.append(
            "Investigar la pérdida de desempeño y evaluar el reentrenamiento."
        )

    if not acciones:
        acciones.append("Continuar el monitoreo sin intervención inmediata.")

    return {
        "nivel": nivel_general,
        "drift_detectado": hay_drift,
        "deterioro_detectado": hay_deterioro,
        "variables_afectadas": tabla.loc[
            tabla["drift_detectado"], "variable"
        ].tolist(),
        "acciones": acciones,
        "detalle": tabla,
    }
