
"""Visualizaciones de errores de predicción del modelo MiBici."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTACIONES_CRITICAS = (3, 71, 158, 266)


def generar_figuras_diagnostico(
    diagnostico: pd.DataFrame,
    directorio_salida: Path,
) -> list[Path]:
    """Genera figuras del diagnóstico utilizando predicciones individuales."""

    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    columnas = {
        "estacion_id",
        "hora",
        "valor_real",
        "prediccion",
        "error_cuadratico",
    }

    faltantes = columnas - set(diagnostico.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    rutas = []

    # 1. RMSE por hora
    errores_hora = (
        diagnostico.groupby("hora")["error_cuadratico"]
        .mean()
        .pow(0.5)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        errores_hora.index.astype(int),
        errores_hora.values,
    )

    ax.set(
        title="HGB Poisson: RMSE por hora, junio de 2026",
        xlabel="Hora del día",
        ylabel="RMSE",
    )

    ax.set_xticks(errores_hora.index.astype(int))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    ruta = directorio_salida / "hgb_rmse_por_hora_junio.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    # 2. Contribución por estación
    contribuciones = (
        diagnostico.groupby("estacion_id")["error_cuadratico"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    contribuciones = (
        contribuciones
        / diagnostico["error_cuadratico"].sum()
        * 100
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        contribuciones.index.astype(str),
        contribuciones.values,
    )

    ax.invert_yaxis()

    ax.set(
        title="HGB Poisson: estaciones con mayor error cuadrático",
        xlabel="Contribución al error cuadrático total (%)",
        ylabel="Estación",
    )

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    ruta = directorio_salida / "hgb_error_por_estacion_junio.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    # 3. Predicciones extremas en estaciones críticas
    afectados = diagnostico.loc[
        diagnostico["estacion_id"].isin(ESTACIONES_CRITICAS)
        & diagnostico["hora"].eq(1)
    ].copy()

    resumen = (
        afectados.groupby("estacion_id")
        .agg(
            real_maximo=("valor_real", "max"),
            prediccion_maxima=("prediccion", "max"),
        )
        .reindex(ESTACIONES_CRITICAS)
    )

    posiciones = np.arange(len(resumen))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        posiciones - ancho / 2,
        resumen["real_maximo"],
        width=ancho,
        label="Demanda real máxima",
    )

    ax.bar(
        posiciones + ancho / 2,
        resumen["prediccion_maxima"],
        width=ancho,
        label="Predicción máxima",
    )

    ax.set_xticks(posiciones, resumen.index.astype(str))

    ax.set(
        title="HGB Poisson: predicciones extremas a la hora 1",
        xlabel="Estación",
        ylabel="Salidas de bicicletas",
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    ruta = directorio_salida / "hgb_predicciones_extremas_junio.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    return rutas
