"""Pruebas de la validación walk-forward, las métricas y la configuración."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mibici.config import cargar_parametros
from mibici.modeling.validation import (
    WalkForwardSplitter,
    evaluar_walk_forward,
    mae,
    rmse,
    wape,
)

MESES_PRUEBA = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]


def datos_sinteticos():
    """Una fila cada 6 h, de octubre de 2025 a septiembre de 2026, con un objetivo lineal."""
    fechas = pd.date_range("2025-10-01", "2026-09-30 23:00", freq="6h")
    x = np.arange(len(fechas), dtype=float)
    X = pd.DataFrame({"fecha_hora": fechas, "x": x})
    y = 2 * x + 1
    return X, y


def modelo_ridge(alpha=1.0):
    """Pipeline que descarta fecha_hora (solo la usa el splitter) y ajusta un Ridge."""
    columnas = ColumnTransformer([("x", "passthrough", ["x"])], remainder="drop")
    return Pipeline([("columnas", columnas), ("modelo", Ridge(alpha=alpha))])


def test_splitter_genera_seis_ventanas_cronologicas():
    X, _ = datos_sinteticos()
    splitter = WalkForwardSplitter(MESES_PRUEBA)

    ventanas = list(splitter.split(X))

    assert len(ventanas) == 6
    assert splitter.get_n_splits() == 6

    for (entrenamiento, prueba), mes in zip(ventanas, MESES_PRUEBA):
        periodo = pd.Period(mes, freq="M")
        fechas_entrenamiento = X["fecha_hora"].iloc[entrenamiento]
        fechas_prueba = X["fecha_hora"].iloc[prueba]

        # Sin traslape y sin datos del mes de prueba en el entrenamiento
        assert set(entrenamiento).isdisjoint(prueba)
        assert fechas_entrenamiento.max() < periodo.start_time
        assert (fechas_prueba.dt.to_period("M") == periodo).all()

    # Ventana creciente: cada entrenamiento contiene al anterior
    tamanos = [len(entrenamiento) for entrenamiento, _ in ventanas]
    assert tamanos == sorted(tamanos)


def test_brecha_quita_horas_antes_del_mes_de_prueba():
    X = pd.DataFrame({"fecha_hora": pd.date_range("2026-03-31 20:00", periods=8, freq="h")})
    splitter = WalkForwardSplitter(["2026-04"], brecha_horas=1)

    entrenamiento, _ = next(splitter.split(X))

    # Se excluye la fila de las 23:00 del 31 de marzo
    assert X["fecha_hora"].iloc[entrenamiento].max() == pd.Timestamp("2026-03-31 22:00")


def test_desde_parametros_usa_params_yaml():
    splitter = WalkForwardSplitter.desde_parametros()

    assert [str(mes) for mes in splitter.meses_prueba] == MESES_PRUEBA
    assert splitter.brecha_horas == 1


def test_splitter_funciona_como_cv_en_gridsearch():
    X, y = datos_sinteticos()
    busqueda = GridSearchCV(
        modelo_ridge(),
        {"modelo__alpha": [0.1, 1.0]},
        cv=WalkForwardSplitter(MESES_PRUEBA),
        scoring="neg_mean_absolute_error",
    )

    busqueda.fit(X, y)

    assert len(busqueda.cv_results_["split5_test_score"]) == 2


def test_metricas_con_valores_conocidos():
    y_real = [1, 2, 3, 4]
    y_pred = [2, 2, 2, 2]

    assert mae(y_real, y_pred) == 1.0
    assert rmse(y_real, y_pred) == pytest.approx(math.sqrt(1.5))
    assert wape(y_real, y_pred) == pytest.approx(0.4)


def test_wape_sin_demanda_regresa_nan():
    assert math.isnan(wape([0, 0], [1, 2]))


def test_evaluar_walk_forward_regresa_tabla_y_resumen():
    X, y = datos_sinteticos()

    tabla, resumen = evaluar_walk_forward(
        modelo_ridge(alpha=0.0), X, y, WalkForwardSplitter(MESES_PRUEBA)
    )

    assert len(tabla) == 6
    assert {"mes_prueba", "n_entrenamiento", "n_prueba", "mae", "rmse", "wape"} <= set(tabla)
    assert set(resumen) == {f"{m}_{e}" for m in ["mae", "rmse", "wape"] for e in ["media", "desv"]}
    # El objetivo es lineal: el error debe ser prácticamente cero
    assert resumen["mae_media"] < 1e-6


def test_cargar_parametros_lee_el_archivo():
    params = cargar_parametros()

    assert {"datos", "limpieza", "features", "validacion", "modelado", "mlflow"} <= set(params)
    assert params["features"]["rezagos"] == [1, 24, 168]
    assert params["features"]["horizonte_horas"] == 1
    assert params["validacion"]["mes_prueba_inicio"] == "2026-04"
    assert params["modelado"]["busqueda"] == {"tipo": "aleatoria", "n_iter": 8}
    assert params["mlflow"]["experimento"] == "mibici-fase2-modelado"
