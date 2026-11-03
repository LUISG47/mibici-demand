"""Pruebas del registro en MLflow con un servidor temporal (no toca mlflow.db)."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from mibici.config import MLFLOW_TRACKING_URI
from mibici.modeling import tracking


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """Apunta tracking.py a un MLflow en tmp_path y restaura la URI al terminar."""
    uri_anterior = mlflow.get_tracking_uri()

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlruns")

    yield uri

    mlflow.set_tracking_uri(uri_anterior)


def pipeline_con_target_encoder():
    """Pipeline pequeño parecido al de modelado: escala un rezago y codifica la estación."""
    preprocesamiento = ColumnTransformer(
        [
            ("historicas", StandardScaler(), ["salidas_lag_1"]),
            (
                "estacion",
                TargetEncoder(
                    target_type="continuous",
                    cv=KFold(n_splits=3, shuffle=True, random_state=42),
                ),
                ["estacion_id"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocesamiento", preprocesamiento), ("modelo", Ridge(alpha=1.0))])


def test_registrar_y_cargar_modelo_champion(mlflow_temporal):
    """Registra el modelo con alias champion y lo carga desde el Registry para predecir."""
    n = 60
    X = pd.DataFrame(
        {
            "estacion_id": np.resize([1, 2, 3], n),
            "fecha_hora": pd.date_range("2026-04-01", periods=n, freq="h"),
            "salidas_lag_1": np.arange(n, dtype=float) % 7,
        }
    )
    y = X["salidas_lag_1"].to_numpy() + X["estacion_id"].to_numpy()

    modelo = pipeline_con_target_encoder().fit(X, y)

    tabla = pd.DataFrame(
        [{"mes_prueba": "2026-04", "n_entrenamiento": 40, "n_prueba": 20, "mae": 0.5, "rmse": 0.7, "wape": 0.1}]
    )
    resumen = {"mae_media": 0.5, "rmse_media": 0.7, "wape_media": 0.1}

    tracking.configurar_mlflow("prueba-tracking")
    assert mlflow.get_tracking_uri() == mlflow_temporal

    run_id = tracking.registrar_evaluacion(
        nombre_modelo="ridge_prueba",
        tabla=tabla,
        resumen=resumen,
        hiperparametros={"algoritmo": "Ridge"},
        modelo_final=modelo,
        X_ejemplo=X.head(5),
    )

    version = tracking.registrar_modelo(
        run_id,
        hiperparametros={"algoritmo": "Ridge"},
        metricas=resumen,
        nombre="modelo-prueba",
        alias="champion",
    )
    assert int(version.version) == 1

    # Cargar desde el Registry y comparar con el modelo en memoria
    cargado = mlflow.sklearn.load_model("models:/modelo-prueba@champion")
    np.testing.assert_allclose(cargado.predict(X.head(10)), modelo.predict(X.head(10)))

    # El registro del proyecto no se usa
    assert mlflow_temporal != MLFLOW_TRACKING_URI
