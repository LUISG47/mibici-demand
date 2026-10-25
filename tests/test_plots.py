"""
Pruebas unitarias para mibici.plots.
"""

import matplotlib
import numpy as np
import pandas as pd

from mibici.plots import graficar_distribucion_duracion_viajes, EDAPlots

# Forzar backend no interactivo para entornos CI/headless
matplotlib.use("Agg")


def test_graficar_distribucion_duracion_viajes():
    durations = pd.Series([0.2, 5.0, 10.5, 15.0, 20.0, 45.0, 120.0])
    fig = graficar_distribucion_duracion_viajes(durations, max_cutoff=60.0)
    assert fig is not None
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Distribución de Duración de Viajes (min)"


def test_edaplots_demanda_hora():
    demanda = pd.Series([100, 200, 150], index=[8, 9, 10])
    fig = EDAPlots.graficar_demanda_por_hora(demanda)
    assert fig is not None
    assert len(fig.axes) == 1


def test_edaplots_demanda_dia_semana():
    df_dia = pd.DataFrame({
        "dia_semana": ["Lunes", "Martes", "Miércoles"],
        "promedio_viajes_dia": [12000.0, 13500.0, 13000.0]
    })
    fig = EDAPlots.graficar_demanda_por_dia_semana(df_dia)
    assert fig is not None
    assert len(fig.axes) == 1


def test_edaplots_mapa_calor_dia_hora():
    matriz = pd.DataFrame(
        np.random.rand(7, 24),
        index=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
        columns=range(24)
    )
    fig = EDAPlots.graficar_mapa_calor_dia_hora(matriz)
    assert fig is not None
    assert len(fig.axes) == 2  # Heatmap genera 2 ejes (ax y colorbar)


def test_edaplots_top_estaciones():
    top_s = pd.DataFrame({"total_salidas": [5000, 4000], "name": ["Est 1", "Est 2"]})
    top_l = pd.DataFrame({"total_llegadas": [4500, 3500], "name": ["Est 3", "Est 4"]})
    fig = EDAPlots.graficar_top_estaciones(top_s, top_l)
    assert fig is not None
    assert len(fig.axes) == 2


def test_edaplots_balance_estaciones():
    df_balance = pd.DataFrame({
        "etiqueta": ["10 - Est 10", "20 - Est 20"],
        "balance_neto": [150, -200]
    })
    fig = EDAPlots.graficar_balance_estaciones(df_balance)
    assert fig is not None
    assert len(fig.axes) == 1


def test_edaplots_distribucion_duracion():
    duraciones = pd.Series([5.0, 10.0, 12.0, 15.0, 20.0])
    fig = EDAPlots.graficar_distribucion_duracion(duraciones, mediana=12.0)
    assert fig is not None
    assert len(fig.axes) == 2


def test_edaplots_distribucion_genero():
    df_genero = pd.DataFrame({
        "genero": ["M", "F"],
        "total_viajes": [3000, 1000],
        "porcentaje": [75.0, 25.0]
    })
    fig = EDAPlots.graficar_distribucion_genero(df_genero)
    assert fig is not None
    assert len(fig.axes) == 1


def test_edaplots_distribucion_edad():
    edades = pd.Series([22, 25, 30, 35, 40, 45])
    fig = EDAPlots.graficar_distribucion_edad(edades)
    assert fig is not None
    assert len(fig.axes) == 1


def test_edaplots_mapa_estaciones():
    df_mapa = pd.DataFrame({
        "longitude": [-103.36, -103.37],
        "latitude": [20.67, 20.68],
        "total_movimientos": [1000, 2500]
    })
    fig = EDAPlots.graficar_mapa_estaciones(df_mapa)
    assert fig is not None
    assert len(fig.axes) == 1