"""
Pruebas unitarias para mibici.plots.
"""

import matplotlib
import pandas as pd
import pytest

from mibici.plots import (
    plot_gender_distribution,
    plot_missing_values,
    plot_monthly_trip_counts,
    plot_trip_duration_distribution,
)

# Forzar backend no interactivo para entornos CI/headless
matplotlib.use("Agg")


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame sintético con columnas completas y con nulos."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "genero": ["M", "F", None, "M", "F"],
            "anio": [1990.0, None, None, 1995.0, 2000.0],
            "valor": [10, 20, 30, 40, 50],
        }
    )


def test_plot_missing_values(sample_dataframe):
    fig = plot_missing_values(sample_dataframe)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    # anio tiene 2 nulos de 5 (40%), debe estar en los ticks del eje Y
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    assert "anio" in y_labels


def test_plot_trip_duration_distribution():
    durations = pd.Series([0.2, 5.0, 10.5, 15.0, 20.0, 45.0, 120.0])
    fig = plot_trip_duration_distribution(durations, max_cutoff=60.0)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    assert ax.get_title() == "Distribución de Duración de Viajes (min)"


def test_plot_monthly_trip_counts():
    monthly_counts = pd.Series(
        [350000, 420000, 380000],
        index=["2025-10", "2025-11", "2025-12"],
    )
    fig = plot_monthly_trip_counts(monthly_counts)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert "2025-10" in x_labels
    assert len(x_labels) == 3


def test_plot_gender_distribution():
    gender_series = pd.Series(
        [3241248, 1223647, 32982],
        index=["M", "F", "NaN"],
    )
    fig = plot_gender_distribution(gender_series)
    assert fig is not None
    assert len(fig.axes) == 1

    ax = fig.axes[0]
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    assert "M" in x_labels
    assert "F" in x_labels
