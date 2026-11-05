"""Pruebas para el módulo de inferencia y predicción."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from typer.testing import CliRunner

from mibici.modeling import predict, tracking
from mibici.modeling.pipeline import crear_pipeline


@pytest.fixture
def mlflow_predict_temporal(tmp_path, monkeypatch):
    """Aísla el Model Registry a una base de datos SQLite temporal."""
    uri_anterior = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlruns")

    yield uri

    mlflow.set_tracking_uri(uri_anterior)


def generar_datos_prueba():
    """Genera datos sintéticos compatibles con el Pipeline v3."""
    fechas = pd.date_range("2026-04-01", periods=10, freq="h")
    datos = pd.DataFrame(
        {
            "estacion_id": [1, 2] * 5,
            "fecha_hora": fechas,
            "fecha_objetivo": fechas + pd.DateOffset(hours=1),
            "salidas_siguiente_hora": np.arange(10) % 3,
            "hora_sin": 0.0,
            "hora_cos": 1.0,
            "dia_semana_sin": 0.0,
            "dia_semana_cos": 1.0,
            "mes_sin": 0.0,
            "mes_cos": 1.0,
            "fin_semana": 0,
            "festivo": 0,
        }
    )
    for lag in [1, 24, 168]:
        datos[f"salidas_lag_{lag}"] = 1.0
        datos[f"llegadas_lag_{lag}"] = 1.0

    return datos


def registrar_modelo_champion_dummy(nombre_modelo="mibici-demanda-salidas", alias="champion"):
    """Entrena y registra un pipeline mínimo con el alias indicado."""
    datos = generar_datos_prueba()
    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    modelo = crear_pipeline(Ridge(alpha=1.0)).fit(X, y)

    tabla = pd.DataFrame(
        [{"mes_prueba": "2026-04", "n_entrenamiento": 6, "n_prueba": 4, "mae": 0.1, "rmse": 0.2, "wape": 0.1}]
    )
    resumen = {"mae_media": 0.1, "rmse_media": 0.2, "wape_media": 0.1}

    tracking.configurar_mlflow("prueba-predict")
    run_id = tracking.registrar_evaluacion(
        nombre_modelo="ridge_test",
        tabla=tabla,
        resumen=resumen,
        hiperparametros={"algoritmo": "Ridge"},
        modelo_final=modelo,
        X_ejemplo=X.head(5),
    )

    tracking.registrar_modelo(
        run_id=run_id,
        hiperparametros={"algoritmo": "Ridge"},
        metricas=resumen,
        nombre=nombre_modelo,
        alias=alias,
    )
    return modelo, X


def test_cargar_modelo_y_predecir(mlflow_predict_temporal):
    """Comprueba que cargar_modelo descargue el champion y predecir genere estimaciones válidas."""
    modelo_original, X = registrar_modelo_champion_dummy()

    modelo_cargado = predict.cargar_modelo(nombre="mibici-demanda-salidas", alias="champion")
    predicciones = predict.predecir(X, modelo=modelo_cargado)

    assert len(predicciones) == len(X)
    assert np.isfinite(predicciones).all()
    np.testing.assert_allclose(predicciones, modelo_original.predict(X))


def test_predecir_carga_modelo_automaticamente(mlflow_predict_temporal):
    """Comprueba que predecir cargue el modelo champion si no se pasa explícitamente."""
    _, X = registrar_modelo_champion_dummy()

    predicciones = predict.predecir(X, modelo=None, nombre="mibici-demanda-salidas", alias="champion")
    assert len(predicciones) == len(X)


def test_predecir_descarta_columna_objetivo(mlflow_predict_temporal):
    """Comprueba que predecir funcione incluso si el DataFrame incluye 'salidas_siguiente_hora'."""
    _, _ = registrar_modelo_champion_dummy()
    datos = generar_datos_prueba()

    predicciones = predict.predecir(datos, nombre="mibici-demanda-salidas", alias="champion")
    assert len(predicciones) == len(datos)


def test_error_cuando_modelo_no_existe(mlflow_predict_temporal):
    """Verifica que se lance una excepción clara si el alias o modelo no existen."""
    tracking.configurar_mlflow("prueba-predict")
    with pytest.raises(Exception):
        predict.cargar_modelo(nombre="modelo-inexistente", alias="champion")


def test_cli_predict(mlflow_predict_temporal, tmp_path):
    """Comprueba la ejecución de inferencia a través de la CLI de Typer."""
    registrar_modelo_champion_dummy()
    datos = generar_datos_prueba()

    entrada_path = tmp_path / "datos_entrada.parquet"
    salida_path = tmp_path / "salida_predicciones.parquet"
    datos.to_parquet(entrada_path, index=False)

    runner = CliRunner()
    resultado = runner.invoke(
        predict.app,
        [
            "--datos-path",
            str(entrada_path),
            "--salida-path",
            str(salida_path),
            "--alias",
            "champion",
            "--modelo-nombre",
            "mibici-demanda-salidas",
        ],
    )

    assert resultado.exit_code == 0
    assert salida_path.exists()

    df_resultado = pd.read_parquet(salida_path)
    assert "salidas_predichas" in df_resultado.columns
    assert "salidas_reales" in df_resultado.columns
    assert len(df_resultado) == len(datos)
