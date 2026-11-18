"""Pruebas automatizadas para el servicio FastAPI de pronóstico de demanda (mibici/api)."""

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from mibici.api import main as api_main
from mibici.api.main import app
from mibici.api.schemas import (
    COLUMNAS_ENTRADA,
    EJEMPLO_REGISTRO_1,
    EJEMPLO_REGISTRO_2,
    SolicitudPrediccion,
)
from mibici.config import PROJ_ROOT

FIXTURES_DIR = PROJ_ROOT / "tests" / "fixtures"


def test_health_ok():
    """GET /health devuelve 200 y confirma que el modelo exportado se cargó en lifespan."""
    with TestClient(app) as client:
        respuesta = client.get("/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelo_cargado"] is True
    assert cuerpo["detalle"] is None


def test_health_y_predict_cuando_falla_carga_modelo(monkeypatch):
    """Si cargar_modelo_servicio falla al iniciar, /health y /predict responden 503 sin tumbar la app."""

    def fallar_carga():
        raise FileNotFoundError("No existe el modelo exportado de prueba")

    monkeypatch.setattr(api_main, "cargar_modelo_servicio", fallar_carga)

    with TestClient(app) as client:
        resp_health = client.get("/health")
        assert resp_health.status_code == 503
        datos_health = resp_health.json()
        assert datos_health["estado"] == "error"
        assert datos_health["modelo_cargado"] is False
        assert "No existe el modelo exportado" in datos_health["detalle"]

        resp_predict = client.post("/predict", json={"registros": [EJEMPLO_REGISTRO_1]})
        assert resp_predict.status_code == 503
        assert "Servicio no disponible" in resp_predict.json()["detail"]


def test_predict_exitoso_individual_y_lote():
    """POST /predict devuelve 200, longitud exacta y predicciones no negativas."""
    payload = {"registros": [EJEMPLO_REGISTRO_1, EJEMPLO_REGISTRO_2]}

    with TestClient(app) as client:
        respuesta = client.post("/predict", json=payload)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["n_predicciones"] == 2
    assert len(cuerpo["salidas_predichas"]) == 2
    assert all(isinstance(valor, float) and valor >= 0.0 for valor in cuerpo["salidas_predichas"])


def test_predict_reproduce_muestra_exportada():
    """POST /predict sobre los primeros registros de muestra_v3_1000.csv reproduce predicciones_v1_muestra.csv."""
    muestra = pd.read_csv(FIXTURES_DIR / "muestra_v3_1000.csv").head(15)
    esperadas = pd.read_csv(
        FIXTURES_DIR / "predicciones_v1_muestra.csv", float_precision="round_trip"
    ).head(15)

    registros = muestra[COLUMNAS_ENTRADA].to_dict(orient="records")

    with TestClient(app) as client:
        respuesta = client.post("/predict", json={"registros": registros})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["n_predicciones"] == 15
    np.testing.assert_allclose(
        cuerpo["salidas_predichas"],
        esperadas["salidas_predichas"].to_numpy(),
        rtol=0,
        atol=1e-9,
    )


def test_predict_admite_rezagos_24_y_168_nulos():
    """POST /predict acepta null en rezagos de 24 h y 168 h (imputados dentro del Pipeline)."""
    registro_con_nulos = {
        **EJEMPLO_REGISTRO_1,
        "salidas_lag_24": None,
        "llegadas_lag_24": None,
        "salidas_lag_168": None,
        "llegadas_lag_168": None,
    }

    with TestClient(app) as client:
        respuesta = client.post("/predict", json={"registros": [registro_con_nulos]})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["n_predicciones"] == 1
    assert cuerpo["salidas_predichas"][0] >= 0.0


def test_solicitud_a_dataframe_respeta_orden_columnas():
    """SolicitudPrediccion.a_dataframe conserva exactamente el orden de COLUMNAS_ENTRADA."""
    solicitud = SolicitudPrediccion(registros=[EJEMPLO_REGISTRO_1])
    df = solicitud.a_dataframe()

    assert list(df.columns) == COLUMNAS_ENTRADA
    assert len(df) == 1


def test_predict_validacion_422_campos_faltantes_y_tipos_invalidos():
    """POST /predict devuelve 422 si faltan columnas requeridas, la lista está vacía o hay tipos inválidos."""
    registro_incompleto = {
        k: v for k, v in EJEMPLO_REGISTRO_1.items() if k not in ("estacion_id", "hora_sin")
    }
    registro_tipo_invalido = {
        **EJEMPLO_REGISTRO_1,
        "estacion_id": "no-es-un-entero",
        "hora": 28.0,  # fuera del rango [0, 23]
    }

    with TestClient(app) as client:
        resp_faltantes = client.post("/predict", json={"registros": [registro_incompleto]})
        resp_tipos = client.post("/predict", json={"registros": [registro_tipo_invalido]})
        resp_vacio = client.post("/predict", json={"registros": []})

    assert resp_faltantes.status_code == 422
    assert resp_tipos.status_code == 422
    assert resp_vacio.status_code == 422


def test_predict_error_interno_inferencia_500(monkeypatch):
    """Si ocurre una excepción durante la inferencia, POST /predict responde 500 con detalle claro."""

    def fallar_predecir(datos, modelo=None):
        raise RuntimeError("Fallo simulado en el pipeline de inferencia")

    monkeypatch.setattr(api_main, "predecir", fallar_predecir)

    with TestClient(app) as client:
        respuesta = client.post("/predict", json={"registros": [EJEMPLO_REGISTRO_1]})

    assert respuesta.status_code == 500
    detalle = respuesta.json()["detail"]
    assert "Error al generar predicciones" in detalle
    assert "Fallo simulado en el pipeline de inferencia" in detalle


def test_docs_swagger_y_openapi_disponibles():
    """GET /docs y GET /openapi.json responden 200 y documentan /health y /predict."""
    with TestClient(app) as client:
        resp_docs = client.get("/docs")
        resp_openapi = client.get("/openapi.json")

    assert resp_docs.status_code == 200
    assert "Swagger UI" in resp_docs.text

    assert resp_openapi.status_code == 200
    esquema = resp_openapi.json()
    assert "/health" in esquema["paths"]
    assert "/predict" in esquema["paths"]
