
"""Detección de data drift mediante PSI y Kolmogórov-Smirnov."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calcular_psi(
    referencia: pd.Series,
    actual: pd.Series,
    n_intervalos: int = 10,
) -> float:
    """Calcula PSI usando intervalos derivados exclusivamente de la referencia."""
    if n_intervalos < 2:
        raise ValueError("n_intervalos debe ser al menos 2.")

    referencia = pd.to_numeric(pd.Series(referencia), errors="coerce")
    actual = pd.to_numeric(pd.Series(actual), errors="coerce")

    if referencia.empty or actual.empty:
        raise ValueError("Las muestras no pueden estar vacías.")

    valores_referencia = referencia.dropna().to_numpy(dtype=float)
    valores_actuales = actual.dropna().to_numpy(dtype=float)

    if not np.isfinite(valores_referencia).all():
        raise ValueError("La referencia contiene valores no finitos.")
    if not np.isfinite(valores_actuales).all():
        raise ValueError("La muestra actual contiene valores no finitos.")

    if len(valores_referencia) == 0:
        raise ValueError("La referencia no contiene valores válidos.")

    # Intervalos calculados únicamente con la distribución de referencia.
    limites = np.unique(
        np.quantile(valores_referencia, np.linspace(0, 1, n_intervalos + 1))
    )

    # Incluir valores fuera del rango de referencia.
    if len(limites) == 1:
        bordes = np.array([-np.inf, limites[0], np.inf])
    else:
        bordes = np.concatenate(([-np.inf], limites[1:-1], [np.inf]))

    conteos_referencia = np.histogram(valores_referencia, bins=bordes)[0]
    conteos_actuales = np.histogram(valores_actuales, bins=bordes)[0]

    # Los valores faltantes forman una categoría adicional.
    conteos_referencia = np.append(conteos_referencia, referencia.isna().sum())
    conteos_actuales = np.append(conteos_actuales, actual.isna().sum())

    # Suavizado para evitar log(0).
    epsilon = 1e-6
    proporcion_referencia = (
        conteos_referencia + epsilon
    ) / (len(referencia) + epsilon * len(conteos_referencia))
    proporcion_actual = (
        conteos_actuales + epsilon
    ) / (len(actual) + epsilon * len(conteos_actuales))

    psi = np.sum(
        (proporcion_actual - proporcion_referencia)
        * np.log(proporcion_actual / proporcion_referencia)
    )

    return float(psi)


def calcular_ks(
    referencia: pd.Series,
    actual: pd.Series,
) -> tuple[float, float]:
    """Regresa el estadístico KS y su valor p para muestras numéricas."""
    referencia = pd.to_numeric(pd.Series(referencia), errors="coerce")
    actual = pd.to_numeric(pd.Series(actual), errors="coerce")

    referencia_valida = referencia.dropna().to_numpy(dtype=float)
    actual_valida = actual.dropna().to_numpy(dtype=float)

    if len(referencia_valida) == 0 or len(actual_valida) == 0:
        raise ValueError("KS requiere valores numéricos válidos en ambas muestras.")

    if not np.isfinite(referencia_valida).all():
        raise ValueError("La referencia contiene valores no finitos.")
    if not np.isfinite(actual_valida).all():
        raise ValueError("La muestra actual contiene valores no finitos.")

    resultado = ks_2samp(
        referencia_valida,
        actual_valida,
        alternative="two-sided",
        method="auto",
    )

    return float(resultado.statistic), float(resultado.pvalue)


def detectar_drift(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    columnas: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Calcula PSI, KS y proporciones de nulos para cada característica."""
    filas = []

    for columna in columnas:
        if columna not in referencia.columns or columna not in actual.columns:
            raise ValueError(f"No existe la columna requerida: {columna}")

        estadistico_ks, pvalor_ks = calcular_ks(
            referencia[columna],
            actual[columna],
        )

        filas.append(
            {
                "variable": columna,
                "psi": calcular_psi(referencia[columna], actual[columna]),
                "ks_estadistico": estadistico_ks,
                "ks_pvalor": pvalor_ks,
                "nulos_referencia": float(referencia[columna].isna().mean()),
                "nulos_actual": float(actual[columna].isna().mean()),
            }
        )

    return pd.DataFrame(filas)
