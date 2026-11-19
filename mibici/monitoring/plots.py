
"""Visualizaciones de diagnóstico para simulaciones de data drift."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from mibici.config import FIGURES_DIR, cargar_parametros


def validar_tablas(
    deteccion: pd.DataFrame,
    desempeno: pd.DataFrame,
) -> None:
    """Comprueba las columnas necesarias para generar las figuras."""
    columnas_deteccion = {
        "escenario",
        "variable",
        "psi",
        "ks_estadistico",
    }
    columnas_desempeno = {
        "escenario",
        "mae",
    }

    faltantes_deteccion = columnas_deteccion - set(deteccion.columns)
    faltantes_desempeno = columnas_desempeno - set(desempeno.columns)

    if faltantes_deteccion:
        raise ValueError(
            f"Faltan columnas de detección: {sorted(faltantes_deteccion)}"
        )

    if faltantes_desempeno:
        raise ValueError(
            f"Faltan columnas de desempeño: {sorted(faltantes_desempeno)}"
        )

    if deteccion.empty or desempeno.empty:
        raise ValueError("Las tablas no pueden estar vacías.")

    if deteccion.duplicated(["escenario", "variable"]).any():
        raise ValueError("Existen variables duplicadas por escenario.")

    if desempeno["escenario"].duplicated().any():
        raise ValueError("Existen escenarios duplicados en desempeño.")


def generar_graficas(
    deteccion: pd.DataFrame,
    desempeno: pd.DataFrame,
    directorio: Path = FIGURES_DIR,
    params: dict | None = None,
) -> dict[str, Path]:
    """Genera gráficas de PSI, KS y desempeño para los escenarios."""
    validar_tablas(deteccion, desempeno)

    if params is None:
        params = cargar_parametros()

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    rutas = {}

    configuraciones = [
        (
            "psi",
            "PSI por variable y escenario",
            "PSI",
            "drift_psi_por_variable.png",
        ),
        (
            "ks_estadistico",
            "KS por variable y escenario",
            "Estadístico KS",
            "drift_ks_por_variable.png",
        ),
    ]

    for columna, titulo, etiqueta, archivo in configuraciones:
        tabla = deteccion.pivot(
            index="variable",
            columns="escenario",
            values=columna,
        )

        figura, eje = plt.subplots(figsize=(12, 6))
        tabla.plot(kind="bar", ax=eje)

        eje.set_title(titulo)
        eje.set_xlabel("Variable")
        eje.set_ylabel(etiqueta)
        eje.tick_params(axis="x", rotation=45)
        eje.legend(title="Escenario")

        if columna == "psi":
            umbrales = params["drift"]["umbrales"]

            eje.axhline(
                float(umbrales["psi_advertencia"]),
                linestyle="--",
                label="Umbral de advertencia",
            )
            eje.axhline(
                float(umbrales["psi_critico"]),
                linestyle=":",
                label="Umbral crítico",
            )
            eje.legend()

        figura.tight_layout()

        ruta = directorio / archivo
        figura.savefig(ruta, dpi=150, bbox_inches="tight")
        plt.close(figura)

        rutas[columna] = ruta

    figura, eje = plt.subplots(figsize=(10, 5))

    desempeno.plot(
        x="escenario",
        y="mae",
        kind="bar",
        legend=False,
        ax=eje,
    )

    referencia = desempeno.loc[
        desempeno["escenario"] == "referencia",
        "mae",
    ]

    if not referencia.empty:
        eje.axhline(
            float(referencia.iloc[0]),
            linestyle="--",
            label="MAE de referencia de la simulación",
        )
        eje.legend()

    eje.set_title("Desempeño del modelo ante data drift")
    eje.set_xlabel("Escenario")
    eje.set_ylabel("MAE")
    eje.tick_params(axis="x", rotation=30)

    figura.tight_layout()

    ruta_mae = directorio / "drift_mae_por_escenario.png"
    figura.savefig(ruta_mae, dpi=150, bbox_inches="tight")
    plt.close(figura)

    rutas["mae"] = ruta_mae

    return rutas
