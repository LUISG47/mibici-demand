"""Visualizaciones de errores de predicción del modelo MiBici."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici.config import cargar_parametros

# Nombres de los meses para títulos y nombres de archivo
MESES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def generar_figuras_diagnostico(
    diagnostico: pd.DataFrame,
    directorio_salida: Path,
    params: dict | None = None,
) -> list[Path]:
    """Genera figuras del diagnóstico utilizando predicciones individuales.

    El mes se toma de fecha_hora, la hora crítica es la de mayor RMSE y las estaciones
    críticas son las que más error cuadrático aportan en esa hora (cuántas, en params.yaml).
    """
    if params is None:
        params = cargar_parametros()

    configuracion = params["analisis_errores"]

    directorio_salida = Path(directorio_salida)
    directorio_salida.mkdir(parents=True, exist_ok=True)

    columnas = {
        "fecha_hora",
        "estacion_id",
        "hora",
        "valor_real",
        "prediccion",
        "error_cuadratico",
    }

    faltantes = columnas - set(diagnostico.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    # Mes evaluado (el más frecuente en fecha_hora)
    mes = pd.to_datetime(diagnostico["fecha_hora"]).dt.to_period("M").mode().iloc[0]
    nombre_mes = MESES[mes.month - 1]

    rutas = []

    # 1. RMSE por hora
    errores_hora = diagnostico.groupby("hora")["error_cuadratico"].mean().pow(0.5).sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        errores_hora.index.astype(int),
        errores_hora.values,
    )

    ax.set(
        title=f"HGB Poisson: RMSE por hora, {nombre_mes} de {mes.year}",
        xlabel="Hora del día",
        ylabel="RMSE",
    )

    ax.set_xticks(errores_hora.index.astype(int))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    ruta = directorio_salida / f"hgb_rmse_por_hora_{nombre_mes}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    # 2. Contribución por estación
    contribuciones = (
        diagnostico.groupby("estacion_id")["error_cuadratico"]
        .sum()
        .sort_values(ascending=False)
        .head(configuracion["top_estaciones"])
    )

    contribuciones = contribuciones / diagnostico["error_cuadratico"].sum() * 100

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

    ruta = directorio_salida / f"hgb_error_por_estacion_{nombre_mes}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    # 3. Predicciones extremas en estaciones críticas
    # Hora crítica: la de mayor RMSE. Estaciones críticas: las de mayor error en esa hora.
    hora_critica = errores_hora.idxmax()
    en_hora = diagnostico.loc[diagnostico["hora"].eq(hora_critica)]

    estaciones_criticas = sorted(
        en_hora.groupby("estacion_id")["error_cuadratico"]
        .sum()
        .nlargest(configuracion["n_estaciones_criticas"])
        .index
    )

    afectados = en_hora.loc[en_hora["estacion_id"].isin(estaciones_criticas)].copy()

    resumen = (
        afectados.groupby("estacion_id")
        .agg(
            real_maximo=("valor_real", "max"),
            prediccion_maxima=("prediccion", "max"),
        )
        .reindex(estaciones_criticas)
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
        title=f"HGB Poisson: predicciones extremas a la hora {int(hora_critica)}",
        xlabel="Estación",
        ylabel="Salidas de bicicletas",
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    ruta = directorio_salida / f"hgb_predicciones_extremas_{nombre_mes}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    rutas.append(ruta)

    return rutas
