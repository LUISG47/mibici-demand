"""
Pruebas unitarias para mibici.plots.
"""

import matplotlib
import pandas as pd

from mibici.plots import (
    graficar_distribucion_duracion_viajes,
    graficar_distribucion_genero,
    graficar_valores_nulos,
    graficar_viajes_mensuales,
)

# Forzar backend no interactivo para entornos CI/headless
matplotlib.use("Agg")


def fixture_dataframe_ejemplo() -> pd.DataFrame:
    """DataFrame sintético con columnas completas y con nulos."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "genero": ["M", "F", None, "M", "F"],
            "anio": [1990.0, None, None, 1995.0, 2000.0],
            "valor": [10, 20, 30, 40, 50],
        }
    )


def test_graficar_valores_nulos():
    datos = fixture_dataframe_ejemplo()
    fig = graficar_valores_nulos(datos)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    # 'anio' tiene 2 nulos de 5 (40%), debe figurar en las etiquetas del eje Y
    etiquetas_y = [label.get_text() for label in ax.get_yticklabels()]
    assert "anio" in etiquetas_y


def test_graficar_distribucion_duracion_viajes():
    duraciones = pd.Series([0.2, 5.0, 10.5, 15.0, 20.0, 45.0, 120.0])
    fig = graficar_distribucion_duracion_viajes(duraciones, max_cutoff=60.0)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    assert ax.get_title() == "Distribución de Duración de Viajes (min)"


def test_graficar_viajes_mensuales():
    conteos_mensuales = pd.Series(
        [350000, 420000, 380000],
        index=["2025-10", "2025-11", "2025-12"],
    )
    fig = graficar_viajes_mensuales(conteos_mensuales)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    etiquetas_x = [label.get_text() for label in ax.get_xticklabels()]
    assert "2025-10" in etiquetas_x
    assert len(etiquetas_x) == 3


def test_graficar_distribucion_genero():
    serie_genero = pd.Series(
        [3241248, 1223647, 32982],
        index=["M", "F", "NaN"],
    )
    fig = graficar_distribucion_genero(serie_genero)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    etiquetas_x = [label.get_text() for label in ax.get_xticklabels()]
    assert "M" in etiquetas_x
    assert "F" in etiquetas_x
