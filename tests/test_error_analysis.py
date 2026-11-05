
"""Pruebas del diagnóstico de errores de HGB Poisson."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.error_analysis import obtener_predicciones_hgb


@pytest.fixture
def datos_prueba():
    return pd.DataFrame(
        {
            "fecha_hora": pd.to_datetime(
                [
                    "2026-05-30 10:00:00",
                    "2026-05-31 10:00:00",
                    "2026-06-01 10:00:00",
                    "2026-06-01 11:00:00",
                ]
            ),
            "estacion_id": [1, 1, 1, 2],
            "hora": [10, 10, 10, 11],
            "salidas_siguiente_hora": [1.0, 2.0, 3.0, 1.0],
        }
    )


@pytest.fixture
def params_prueba():
    return {
        "validacion": {
            "mes_prueba_inicio": "2026-06",
            "mes_prueba_fin": "2026-06",
        },
        "features": {
            "horizonte_horas": 1,
        },
        "modelado": {
            "semilla": 42,
            "hiperparametros": {
                "hist_gradient_boosting": {
                    "loss": ["poisson"],
                    "learning_rate": [0.1],
                    "max_iter": [500],
                    "max_leaf_nodes": [63],
                    "min_samples_leaf": [20],
                },
            },
        },
    }


@pytest.fixture
def checkpoint_prueba():
    return {
        "resultado": {
            "ventana": 1,
            "mes_prueba": "2026-06",
            "n_entrenamiento": 2,
            "n_prueba": 2,
            "loss": "poisson",
            "learning_rate": 0.1,
            "max_iter": 500,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "mae": 1.0,
            "rmse": 1.0,
            "wape": 0.5,
        }
    }


def test_whenPredictionsMatchCheckpoint_thenReturnsIndividualErrors(
    monkeypatch,
    tmp_path,
    datos_prueba,
    params_prueba,
    checkpoint_prueba,
):
    llamadas = {}

    class ModeloSimulado:
        def fit(self, X, y):
            llamadas["entrenamiento"] = X["fecha_hora"].tolist()
            llamadas["objetivo_entrenamiento"] = list(y)
            return self

        def predict(self, X):
            llamadas["prueba"] = X["fecha_hora"].tolist()
            return np.array([2.0, 2.0])

    def crear_pipeline_simulado(estimador, params):
        llamadas["estimador"] = estimador
        return ModeloSimulado()

    monkeypatch.setattr(
        "mibici.modeling.error_analysis.cargar_checkpoint",
        lambda ruta: checkpoint_prueba,
    )

    monkeypatch.setattr(
        "mibici.modeling.error_analysis.crear_pipeline",
        crear_pipeline_simulado,
    )

    diagnostico, metricas = obtener_predicciones_hgb(
        datos_prueba,
        ruta_checkpoint=tmp_path / "resultado.json",
        params=params_prueba,
    )

    assert len(llamadas["entrenamiento"]) == 2
    assert len(llamadas["prueba"]) == 2

    assert max(llamadas["entrenamiento"]) < min(
        llamadas["prueba"]
    )

    assert llamadas["estimador"].loss == "poisson"
    assert llamadas["estimador"].max_iter == 500
    assert llamadas["estimador"].random_state == 42

    assert diagnostico["valor_real"].tolist() == [3.0, 1.0]
    assert diagnostico["prediccion"].tolist() == [2.0, 2.0]
    assert diagnostico["error"].tolist() == [1.0, -1.0]
    assert diagnostico["error_absoluto"].tolist() == [1.0, 1.0]
    assert diagnostico["error_cuadratico"].tolist() == [1.0, 1.0]

    assert diagnostico["estacion_id"].tolist() == [1, 2]
    assert diagnostico["hora"].tolist() == [10, 11]

    assert metricas == {
        "mae": 1.0,
        "rmse": 1.0,
        "wape": 0.5,
    }


def test_whenCheckpointIsMissing_thenRaisesFileNotFoundError(
    tmp_path,
    datos_prueba,
    params_prueba,
):
    with pytest.raises(FileNotFoundError, match="No existe"):
        obtener_predicciones_hgb(
            datos_prueba,
            ruta_checkpoint=tmp_path / "inexistente.json",
            params=params_prueba,
        )


def test_whenCheckpointMetricDiffers_thenRaisesValueError(
    monkeypatch,
    tmp_path,
    datos_prueba,
    params_prueba,
    checkpoint_prueba,
):
    checkpoint_prueba["resultado"]["rmse"] = 57.0

    class ModeloSimulado:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([2.0, 2.0])

    monkeypatch.setattr(
        "mibici.modeling.error_analysis.cargar_checkpoint",
        lambda ruta: checkpoint_prueba,
    )

    monkeypatch.setattr(
        "mibici.modeling.error_analysis.crear_pipeline",
        lambda estimador, params: ModeloSimulado(),
    )

    with pytest.raises(ValueError, match="métrica rmse no coincide"):
        obtener_predicciones_hgb(
            datos_prueba,
            ruta_checkpoint=tmp_path / "resultado.json",
            params=params_prueba,
        )
