
"""Registro de experimentos de data drift en MLflow."""

import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mibici.config import cargar_parametros
from mibici.modeling.tracking import configurar_mlflow


EXPERIMENTO_DRIFT = "mibici-fase2-drift"


def registrar_drift(
    escenario: str,
    deteccion: pd.DataFrame,
    metricas: dict,
    alertas: dict,
    params: dict | None = None,
) -> str:
    """Registra un escenario de drift y devuelve el run_id de MLflow."""
    if params is None:
        params = cargar_parametros()

    if deteccion.empty:
        raise ValueError("La tabla de detección no puede estar vacía.")

    requeridas = {"mae", "rmse", "wape", "deterioro_mae_relativo"}
    faltantes = requeridas - set(metricas)

    if faltantes:
        raise ValueError(f"Faltan métricas: {sorted(faltantes)}")

    configuracion = params["drift"]

    # Reutilizar el tracking existente sin modificar sus funciones.
    configurar_mlflow(EXPERIMENTO_DRIFT)

    with mlflow.start_run(run_name=f"drift_{escenario}") as run:
        mlflow.log_params(
            {
                "escenario": escenario,
                "mes_referencia": configuracion["mes_referencia"],
                "semilla": configuracion["semilla"],
                "mae_referencia_historica": configuracion["mae_referencia"],
            }
        )

        for nombre, valor in configuracion["umbrales"].items():
            mlflow.log_param(f"umbral_{nombre}", valor)

        for nombre in sorted(requeridas):
            valor = float(metricas[nombre])
            if np.isfinite(valor):
                mlflow.log_metric(nombre, valor)

        mlflow.log_metric(
            "variables_con_drift",
            len(alertas["variables_afectadas"]),
        )

        mlflow.set_tags(
            {
                "tipo_experimento": "simulacion_data_drift",
                "nivel_alerta": alertas["nivel"],
                "drift_detectado": str(alertas["drift_detectado"]).lower(),
                "deterioro_detectado": str(
                    alertas["deterioro_detectado"]
                ).lower(),
            }
        )

        with tempfile.TemporaryDirectory() as directorio:
            carpeta = Path(directorio)

            ruta_deteccion = carpeta / "deteccion_variables.csv"
            deteccion.to_csv(ruta_deteccion, index=False)
            mlflow.log_artifact(
                str(ruta_deteccion),
                artifact_path="drift",
            )

            ruta_alertas = carpeta / "alertas.json"
            resumen = {
                "escenario": escenario,
                "nivel": alertas["nivel"],
                "drift_detectado": alertas["drift_detectado"],
                "deterioro_detectado": alertas["deterioro_detectado"],
                "variables_afectadas": alertas["variables_afectadas"],
                "acciones": alertas["acciones"],
            }

            ruta_alertas.write_text(
                json.dumps(resumen, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(
                str(ruta_alertas),
                artifact_path="drift",
            )

    return run.info.run_id
