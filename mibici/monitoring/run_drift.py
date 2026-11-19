
"""Ejecución integrada del monitoreo de data drift para MiBici."""

from pathlib import Path

import pandas as pd
from loguru import logger

from mibici.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    cargar_parametros,
)
from mibici.modeling.predict import cargar_modelo_servicio
from mibici.monitoring.alerts import evaluar_alertas
from mibici.monitoring.detectors import detectar_drift
from mibici.monitoring.drift import (
    COLUMNAS_HISTORICAS,
    ESCENARIOS,
    simular_drift,
)
from mibici.monitoring.evaluation import evaluar_escenarios
from mibici.monitoring.plots import generar_graficas
from mibici.monitoring.tracking import registrar_drift


def ejecutar_monitoreo(
    datos_path: Path = PROCESSED_DATA_DIR / "estacion_hora_v3.parquet",
    figuras_dir: Path = FIGURES_DIR,
    params: dict | None = None,
) -> dict:
    """Ejecuta la evaluación de drift sobre el mes de referencia."""
    if params is None:
        params = cargar_parametros()

    datos_path = Path(datos_path)

    if not datos_path.is_file():
        raise FileNotFoundError(f"No existe el dataset: {datos_path}")

    logger.info(f"Cargando dataset: {datos_path}")
    datos = pd.read_parquet(datos_path)

    mes = params["drift"]["mes_referencia"]
    fechas = pd.to_datetime(datos["fecha_hora"])

    referencia = datos.loc[
        fechas.dt.to_period("M") == mes
    ].copy()

    if referencia.empty:
        raise ValueError(f"No existen registros para {mes}.")

    logger.info(
        f"Evaluando {len(referencia):,} observaciones de {mes}."
    )

    modelo = cargar_modelo_servicio(params=params)
    desempeno = evaluar_escenarios(referencia, modelo, params=params)

    tablas_deteccion = []
    resumen_alertas = []
    run_ids = {}

    for escenario in ESCENARIOS:
        logger.info(f"Analizando escenario: {escenario}")

        alterados = simular_drift(
            referencia,
            escenario,
            params=params,
        )

        deteccion = detectar_drift(
            referencia,
            alterados,
            list(COLUMNAS_HISTORICAS),
        )

        deteccion.insert(0, "escenario", escenario)

        fila = desempeno.loc[
            desempeno["escenario"] == escenario
        ].iloc[0]

        metricas = {
            "mae": float(fila["mae"]),
            "rmse": float(fila["rmse"]),
            "wape": float(fila["wape"]),
            "deterioro_mae_relativo": float(
                fila["deterioro_mae_relativo"]
            ),
        }

        alertas = evaluar_alertas(
            deteccion,
            metricas["deterioro_mae_relativo"],
            params=params,
        )

        run_id = registrar_drift(
            escenario=escenario,
            deteccion=deteccion,
            metricas=metricas,
            alertas=alertas,
            params=params,
        )

        run_ids[escenario] = run_id
        tablas_deteccion.append(deteccion)

        resumen_alertas.append(
            {
                "escenario": escenario,
                "nivel": alertas["nivel"],
                "drift_detectado": alertas["drift_detectado"],
                "deterioro_detectado": alertas[
                    "deterioro_detectado"
                ],
                "variables_afectadas": len(
                    alertas["variables_afectadas"]
                ),
                "acciones": " | ".join(alertas["acciones"]),
            }
        )

    deteccion_total = pd.concat(
        tablas_deteccion,
        ignore_index=True,
    )

    figuras = generar_graficas(
        deteccion_total,
        desempeno,
        directorio=figuras_dir,
        params=params,
    )

    return {
        "desempeno": desempeno,
        "deteccion": deteccion_total,
        "alertas": pd.DataFrame(resumen_alertas),
        "figuras": figuras,
        "run_ids": run_ids,
    }


if __name__ == "__main__":
    resultado = ejecutar_monitoreo()

    print("\nDESEMPEÑO POR ESCENARIO")
    print(resultado["desempeno"].to_string(index=False))

    print("\nALERTAS")
    print(resultado["alertas"].to_string(index=False))

    print("\nFIGURAS GENERADAS")
    for nombre, ruta in resultado["figuras"].items():
        print(f"{nombre}: {ruta}")

    print("\nRUNS DE MLFLOW")
    for escenario, run_id in resultado["run_ids"].items():
        print(f"{escenario}: {run_id}")
