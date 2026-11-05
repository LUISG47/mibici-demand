"""Pruebas de la comparación y registro de modelos en MLflow."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from mibici.modeling.baselines import HistoricalMeanBaseline, WeeklyLagBaseline
from mibici.modeling.compare_models import (
    comparar_modelos,
    crear_modelo_final,
    hiperparametros_frecuentes,
)


@pytest.fixture
def params_prueba():
    return {
        "mlflow": {"experimento": "experimento_prueba"},
        "features": {
            "horizonte_horas": 1,
            "rezagos": [1, 24, 168],
        },
        "modelado": {
            "semilla": 42,
            "target_encoder_cv": 2,
            "hiperparametros": {
                "ridge": {"alpha": [10.0, 100.0]},
                "hist_gradient_boosting": {
                    "loss": ["poisson"],
                    "max_iter": [200, 500],
                },
            },
        },
    }


def resultado(mae, **hiperparametros):
    """Tabla de una ventana y resumen simulados."""
    tabla = pd.DataFrame(
        {
            "mes_prueba": ["2026-04"],
            "n_entrenamiento": [100],
            "n_prueba": [20],
            **{clave: [valor] for clave, valor in hiperparametros.items()},
            "mae": [mae],
            "rmse": [mae + 0.5],
            "wape": [mae / 2],
        }
    )

    resumen = {
        "mae_media": mae,
        "mae_desv": 0.0,
        "rmse_media": mae + 0.5,
        "rmse_desv": 0.0,
        "wape_media": mae / 2,
        "wape_desv": 0.0,
    }

    return tabla, resumen


def test_whenModelsAreCompared_thenRegistersFourEvaluationsAndChampion(
    monkeypatch, tmp_path, params_prueba
):
    """Registra cuatro modelos y solo el mejor con modelo final y alias champion."""
    datos = pd.DataFrame(
        {
            "fecha_hora": [pd.Timestamp("2026-04-01")],
            "estacion_id": [1],
            "salidas_siguiente_hora": [2.0],
        }
    )

    llamadas = []
    registros = []
    registro_modelo = []

    def evaluar_baselines_simulado(datos, params):
        llamadas.append("baselines")

        tabla_semanal, resumen_semanal = resultado(0.9)
        tabla_historico, resumen_historico = resultado(0.8)

        return {
            "baseline_lag_168": {"tabla": tabla_semanal, "resumen": resumen_semanal},
            "baseline_media_historica": {
                "tabla": tabla_historico,
                "resumen": resumen_historico,
            },
        }

    def evaluar_ridge_simulado(datos, params):
        llamadas.append("ridge")
        return resultado(0.7, alpha=100.0)

    def evaluar_hgb_simulado(datos, params, checkpoint_dir):
        llamadas.append("hgb")
        assert Path(checkpoint_dir) == tmp_path / "checkpoints_hgb"
        return resultado(0.6, loss="poisson", max_iter=500)

    class ModeloSimulado:
        def __init__(self, nombre, hiperparametros):
            self.nombre = nombre
            self.hiperparametros = hiperparametros

        def fit(self, X, y):
            llamadas.append(f"fit_{self.nombre}")
            assert "salidas_siguiente_hora" not in X.columns
            return self

    def crear_modelo_final_simulado(nombre, hiperparametros, params):
        return ModeloSimulado(nombre, hiperparametros)

    def configurar_mlflow_simulado(experimento):
        llamadas.append("configurar_mlflow")
        assert experimento == "experimento_prueba"

    def registrar_simulado(
        nombre_modelo, tabla, resumen, hiperparametros, modelo_final=None, X_ejemplo=None
    ):
        registros.append(
            {
                "nombre": nombre_modelo,
                "tabla": tabla,
                "hiperparametros": hiperparametros,
                "modelo_final": modelo_final,
                "X_ejemplo": X_ejemplo,
            }
        )
        return f"run_{nombre_modelo}"

    def registrar_modelo_simulado(run_id, hiperparametros, metricas, alias=None):
        registro_modelo.append(
            {"run_id": run_id, "hiperparametros": hiperparametros, "alias": alias}
        )

        class Version:
            version = "1"

        return Version()

    modulo = "mibici.modeling.compare_models"
    monkeypatch.setattr(f"{modulo}.evaluar_baselines", evaluar_baselines_simulado)
    monkeypatch.setattr(f"{modulo}.evaluar_ridge_ajustado", evaluar_ridge_simulado)
    monkeypatch.setattr(f"{modulo}.evaluar_hgb_ajustado", evaluar_hgb_simulado)
    monkeypatch.setattr(f"{modulo}.crear_modelo_final", crear_modelo_final_simulado)
    monkeypatch.setattr(f"{modulo}.configurar_mlflow", configurar_mlflow_simulado)
    monkeypatch.setattr(f"{modulo}.registrar_evaluacion", registrar_simulado)
    monkeypatch.setattr(f"{modulo}.registrar_modelo", registrar_modelo_simulado)

    comparacion = comparar_modelos(
        datos,
        params=params_prueba,
        checkpoint_dir=tmp_path / "checkpoints_hgb",
    )

    # Solo se entrena el modelo final del mejor (HGB) antes de registrar
    assert llamadas == ["baselines", "ridge", "hgb", "fit_hgb_poisson", "configurar_mlflow"]

    assert len(registros) == 4
    assert comparacion["modelo"].tolist() == [
        "hgb_poisson",
        "ridge_ajustado",
        "baseline_media_historica",
        "baseline_lag_168",
    ]
    assert comparacion["mae_media"].tolist() == [0.6, 0.7, 0.8, 0.9]

    por_nombre = {registro["nombre"]: registro for registro in registros}

    for registro in registros:
        assert registro["hiperparametros"]["validacion"] == "walk_forward"
        assert registro["hiperparametros"]["horizonte_horas"] == 1
        assert registro["hiperparametros"]["rezagos"] == "1,24,168"
        assert len(registro["tabla"]) == 1

    # Hiperparámetros elegidos como params del run padre
    assert por_nombre["ridge_ajustado"]["hiperparametros"]["alpha"] == 100.0
    assert por_nombre["hgb_poisson"]["hiperparametros"]["max_iter"] == 500
    assert "alpha" not in por_nombre["baseline_lag_168"]["hiperparametros"]

    # Solo el mejor lleva modelo final y ejemplo de entrada
    assert por_nombre["hgb_poisson"]["modelo_final"].nombre == "hgb_poisson"
    assert por_nombre["hgb_poisson"]["X_ejemplo"] is not None
    for nombre in ["ridge_ajustado", "baseline_media_historica", "baseline_lag_168"]:
        assert por_nombre[nombre]["modelo_final"] is None

    assert registro_modelo == [
        {
            "run_id": "run_hgb_poisson",
            "hiperparametros": por_nombre["hgb_poisson"]["hiperparametros"],
            "alias": "champion",
        }
    ]


def test_whenWindowsDisagree_thenReturnsMostFrequentCombination(params_prueba):
    tabla = pd.DataFrame(
        {
            "loss": ["poisson"] * 3,
            "max_iter": [200, 500, 500],
        }
    )

    elegidos = hiperparametros_frecuentes("hgb_poisson", tabla, params_prueba)

    assert elegidos == {"loss": "poisson", "max_iter": 500}
    assert type(elegidos["max_iter"]) is int


def test_whenFrequenciesTie_thenMostRecentWindowWins(params_prueba):
    tabla = pd.DataFrame({"alpha": [10.0, 100.0]})

    elegidos = hiperparametros_frecuentes("ridge_ajustado", tabla, params_prueba)

    assert elegidos == {"alpha": 100.0}


def test_whenModelIsBaseline_thenHasNoHyperparameters(params_prueba):
    tabla = pd.DataFrame({"mae": [1.0]})

    assert hiperparametros_frecuentes("baseline_lag_168", tabla, params_prueba) == {}


def test_whenBuildingFinalModels_thenUsesExpectedEstimators(params_prueba):
    assert isinstance(crear_modelo_final("baseline_lag_168", {}, params_prueba), WeeklyLagBaseline)
    assert isinstance(
        crear_modelo_final("baseline_media_historica", {}, params_prueba),
        HistoricalMeanBaseline,
    )

    ridge = crear_modelo_final("ridge_ajustado", {"alpha": 100.0}, params_prueba)
    assert isinstance(ridge.named_steps["modelo"], Ridge)
    assert ridge.named_steps["modelo"].alpha == 100.0

    hgb = crear_modelo_final(
        "hgb_poisson", {"loss": "poisson", "max_iter": 500.0}, params_prueba
    )
    estimador = hgb.named_steps["modelo"]
    assert isinstance(estimador, HistGradientBoostingRegressor)
    assert estimador.max_iter == 500
    assert type(estimador.max_iter) is int
    assert estimador.random_state == 42


def test_whenModelIsUnknown_thenRaisesValueError(params_prueba):
    with pytest.raises(ValueError, match="Modelo desconocido"):
        crear_modelo_final("otro", {}, params_prueba)
