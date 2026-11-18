"""Aplicación FastAPI para servir el modelo de pronóstico de demanda de MiBici."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from mlflow.exceptions import MlflowException

from mibici.api.schemas import (
    RespuestaPrediccion,
    RespuestaSalud,
    SolicitudPrediccion,
)
from mibici.config import cargar_parametros
from mibici.modeling.predict import cargar_modelo_servicio, predecir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo MLflow una sola vez al iniciar el servicio."""
    app.state.modelo = None
    app.state.error_modelo = None

    try:
        logger.info("Iniciando carga del modelo para el servicio FastAPI...")
        app.state.modelo = cargar_modelo_servicio()
        logger.success("Modelo cargado exitosamente en app.state.modelo.")
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        MlflowException,
        OSError,
        RuntimeError,
    ) as exc:
        logger.error(f"Error al cargar el modelo durante el inicio del servicio: {exc}")
        app.state.error_modelo = str(exc)

    yield

    app.state.modelo = None
    app.state.error_modelo = None


def crear_app() -> FastAPI:
    """Construye y configura la instancia de FastAPI usando params.yaml."""
    api_cfg = cargar_parametros().get("api", {})

    return FastAPI(
        title=api_cfg.get("titulo", "MiBici Demand API"),
        description=api_cfg.get(
            "descripcion",
            "Servicio de pronóstico de salidas de bicicletas por estación y hora (horizonte 1 h).",
        ),
        version=str(api_cfg.get("version", "1.0.0")),
        lifespan=lifespan,
    )


app = crear_app()


@app.get(
    "/health",
    response_model=RespuestaSalud,
    tags=["Salud"],
    summary="Verifica el estado del servicio y del modelo cargado",
    responses={
        status.HTTP_200_OK: {"description": "Servicio activo y modelo listo."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "El modelo no pudo cargarse o no está disponible."
        },
    },
)
def health(request: Request):
    """Reporta la salud del servicio para el HEALTHCHECK de Docker."""
    modelo = getattr(request.app.state, "modelo", None)
    error_modelo = getattr(request.app.state, "error_modelo", None)

    if modelo is None:
        contenido = RespuestaSalud(
            estado="error",
            modelo_cargado=False,
            detalle=error_modelo or "El modelo no está cargado.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=contenido.model_dump(),
        )

    return RespuestaSalud(
        estado="ok",
        modelo_cargado=True,
        detalle=None,
    )


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    tags=["Inferencia"],
    summary="Pronostica las salidas de bicicletas en la siguiente hora",
    responses={
        status.HTTP_200_OK: {"description": "Predicciones generadas exitosamente."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Error de validación en los campos o tipos de entrada."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Error interno durante la ejecución de la inferencia."
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "El modelo no está cargado en el servicio."
        },
    },
)
def predict_endpoint(solicitud: SolicitudPrediccion, request: Request) -> RespuestaPrediccion:
    """Valida los registros de entrada, construye el DataFrame y ejecuta el modelo."""
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        detalle = getattr(request.app.state, "error_modelo", None) or "El modelo no está cargado."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Servicio no disponible: {detalle}",
        )

    try:
        datos = solicitud.a_dataframe()
        predicciones = predecir(datos, modelo=modelo)
        lista_predicciones = [float(valor) for valor in predicciones]
    except Exception as exc:
        logger.error(f"Error durante la inferencia en /predict: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar predicciones: {exc}",
        ) from exc

    return RespuestaPrediccion(
        salidas_predichas=lista_predicciones,
        n_predicciones=len(lista_predicciones),
    )
