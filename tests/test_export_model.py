"""Pruebas de la exportación del modelo a models/ (mibici/modeling/export_model.py)."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from typer.testing import CliRunner

from mibici.config import MODELS_DIR, PROJ_ROOT, cargar_parametros
from mibici.modeling import export_model, predict, tracking
from mibici.modeling.pipeline import crear_pipeline

NOMBRE = "mibici-demanda-salidas"
FIXTURES = PROJ_ROOT / "tests" / "fixtures"


@pytest.fixture
def mlflow_temporal(tmp_path, monkeypatch):
    """Aísla el tracking y el Model Registry en una base SQLite temporal."""
    uri_anterior = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(export_model, "MLFLOW_TRACKING_URI", uri)

    yield uri

    mlflow.set_tracking_uri(uri_anterior)


def generar_datos():
    """Datos sintéticos con las columnas del dataset v3."""
    fechas = pd.date_range("2026-04-01", periods=12, freq="h")
    datos = pd.DataFrame(
        {
            "estacion_id": [1, 2, 3] * 4,
            "fecha_hora": fechas,
            "fecha_objetivo": fechas + pd.DateOffset(hours=1),
            "hora_sin": np.sin(np.arange(12)),
            "hora_cos": np.cos(np.arange(12)),
            "dia_semana_sin": 0.0,
            "dia_semana_cos": 1.0,
            "mes_sin": 0.0,
            "mes_cos": 1.0,
            "fin_semana": 0.0,
            "festivo": 0.0,
        }
    )
    for lag in [1, 24, 168]:
        datos[f"salidas_lag_{lag}"] = np.arange(12) % 4 + 1.0
        datos[f"llegadas_lag_{lag}"] = np.arange(12) % 3 + 1.0
    return datos, np.arange(12) % 5


def registrar_modelo_dummy():
    """Entrena un Pipeline mínimo, lo registra como versión 1 y regresa (modelo, X, run_id)."""
    X, y = generar_datos()
    modelo = crear_pipeline(Ridge(alpha=1.0)).fit(X, y)

    tabla = pd.DataFrame(
        [{"mes_prueba": "2026-04", "n_entrenamiento": 8, "n_prueba": 4,
          "mae": 0.5, "rmse": 0.7, "wape": 0.4}]
    )  # fmt: skip
    resumen = {"mae_media": 0.5, "rmse_media": 0.7, "wape_media": 0.4}

    tracking.configurar_mlflow("prueba-export")
    run_id = tracking.registrar_evaluacion(
        nombre_modelo="ridge_test",
        tabla=tabla,
        resumen=resumen,
        hiperparametros={"algoritmo": "ridge", "alpha": 1.0},
        modelo_final=modelo,
        X_ejemplo=X,
    )
    tracking.registrar_modelo(
        run_id, {"algoritmo": "ridge"}, resumen, nombre=NOMBRE, alias="champion"
    )
    return modelo, X, run_id


def test_exportar_modelo_crea_modelo_mlflow_y_metadata(mlflow_temporal, tmp_path):
    """La carpeta exportada es un modelo MLflow con metadata.json trazable al Registry."""
    _, X, run_id = registrar_modelo_dummy()
    destino = tmp_path / "exportado"

    ruta = export_model.exportar_modelo(nombre=NOMBRE, version=1, destino=destino)

    assert ruta == destino
    for archivo in ["MLmodel", "model.pkl", "requirements.txt", "metadata.json"]:
        assert (destino / archivo).exists(), archivo

    metadata = json.loads((destino / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["modelo_uri"] == f"models:/{NOMBRE}/1"
    assert metadata["version"] == 1
    assert metadata["run_id"] == run_id
    assert metadata["aliases"] == ["champion"]
    assert metadata["metricas_promedio"]["mae_media"] == pytest.approx(0.5)
    assert metadata["hiperparametros"]["algoritmo"] == "ridge"
    assert metadata["columnas_entrada"] == list(X.columns)
    assert metadata["sha256_model_pkl"] == export_model.sha256_archivo(destino / "model.pkl")
    for clave in ["git_commit", "git_commit_exportacion", "fecha_exportacion"]:
        assert metadata[clave]


def test_modelo_exportado_predice_igual_que_el_registry(mlflow_temporal, tmp_path):
    """El modelo cargado desde la carpeta exportada da las mismas predicciones."""
    modelo, X, _ = registrar_modelo_dummy()
    destino = export_model.exportar_modelo(nombre=NOMBRE, version=1, destino=tmp_path / "m")

    exportado = predict.cargar_modelo_exportado(destino)

    np.testing.assert_array_equal(exportado.predict(X), modelo.predict(X))


def test_exportar_no_sobrescribe_sin_permiso(mlflow_temporal, tmp_path):
    """Si el destino ya existe, falla salvo que se pida sobrescribir."""
    registrar_modelo_dummy()
    destino = tmp_path / "m"
    export_model.exportar_modelo(nombre=NOMBRE, version=1, destino=destino)

    with pytest.raises(FileExistsError):
        export_model.exportar_modelo(nombre=NOMBRE, version=1, destino=destino)

    export_model.exportar_modelo(nombre=NOMBRE, version=1, destino=destino, sobrescribir=True)
    assert (destino / "metadata.json").exists()


def test_cli_exportar(mlflow_temporal, tmp_path):
    """El CLI de typer exporta el modelo a la carpeta indicada."""
    registrar_modelo_dummy()
    destino = tmp_path / "cli"

    resultado = CliRunner().invoke(
        export_model.app, ["--nombre", NOMBRE, "--version", "1", "--destino", str(destino)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert (destino / "metadata.json").exists()


def test_modelo_v1_versionado_reproduce_predicciones():
    """El modelo versionado en models/ reproduce las predicciones de predict.py (1e-9).

    Las predicciones esperadas se generaron con predict.py local desde el Registry
    (models:/mibici-demanda-salidas@champion, versión 1) sobre la muestra fija.
    """
    serving = cargar_parametros()["serving"]
    ruta = MODELS_DIR / serving["directorio_modelo"]
    metadata = json.loads((ruta / "metadata.json").read_text(encoding="utf-8"))
    assert export_model.sha256_archivo(ruta / "model.pkl") == metadata["sha256_model_pkl"]

    muestra = pd.read_csv(FIXTURES / "muestra_v3_1000.csv")
    esperadas = pd.read_csv(FIXTURES / "predicciones_v1_muestra.csv", float_precision="round_trip")

    predicciones = predict.predecir(muestra, modelo=predict.cargar_modelo_servicio())

    assert len(predicciones) == 1000
    np.testing.assert_allclose(predicciones, esperadas["salidas_predichas"], rtol=0, atol=1e-9)
