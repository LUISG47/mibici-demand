
"""Simulación reproducible de data drift para MiBici."""

import numpy as np
import pandas as pd

from mibici.config import cargar_parametros


COLUMNAS_HISTORICAS = (
    "salidas_lag_1",
    "salidas_lag_24",
    "salidas_lag_168",
    "llegadas_lag_1",
    "llegadas_lag_24",
    "llegadas_lag_168",
)

ESCENARIOS = (
    "desplazamiento_medias",
    "features_faltantes",
    "cambio_estacional",
)


def validar_columnas(datos: pd.DataFrame) -> None:
    """Comprueba que existan las características necesarias."""
    requeridas = set(COLUMNAS_HISTORICAS) | {"fin_semana", "festivo"}
    faltantes = sorted(requeridas - set(datos.columns))

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def simular_drift(
    datos: pd.DataFrame,
    escenario: str,
    params: dict | None = None,
) -> pd.DataFrame:
    """Genera un escenario sintético sin modificar los datos originales."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario no soportado: {escenario}")

    validar_columnas(datos)

    if params is None:
        params = cargar_parametros()

    configuracion = params["drift"]
    simulacion = configuracion["simulacion"][escenario]

    resultado = datos.copy(deep=True)
    columnas = list(COLUMNAS_HISTORICAS)

    if escenario == "desplazamiento_medias":
        factor = float(simulacion["factor"])

        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("El factor debe ser positivo y finito.")

        resultado.loc[:, columnas] = resultado[columnas].mul(factor)

    elif escenario == "features_faltantes":
        proporcion = float(simulacion["proporcion"])

        if not 0 <= proporcion <= 1 or not np.isfinite(proporcion):
            raise ValueError("La proporción debe estar entre 0 y 1.")

        generador = np.random.default_rng(int(configuracion["semilla"]))
        mascara = generador.random((len(resultado), len(columnas))) < proporcion

        resultado.loc[:, columnas] = resultado[columnas].mask(mascara)

    elif escenario == "cambio_estacional":
        factor = float(simulacion["factor"])

        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("El factor debe ser positivo y finito.")

        dias_especiales = (
            resultado["fin_semana"].eq(1) | resultado["festivo"].eq(1)
        )

        resultado.loc[dias_especiales, columnas] = (
            resultado.loc[dias_especiales, columnas].mul(factor)
        )

    return resultado
