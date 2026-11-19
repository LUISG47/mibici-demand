"""Esquemas Pydantic de entrada y salida para el servicio de predicción de MiBici."""

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mibici.config import cargar_parametros

# Orden exacto de las 20 columnas de la firma del modelo (models/mibici-demanda-salidas-v1/metadata.json)
COLUMNAS_ENTRADA: list[str] = [
    "estacion_id",
    "fecha_hora",
    "hora",
    "dia_semana",
    "mes",
    "fin_semana",
    "salidas_lag_1",
    "llegadas_lag_1",
    "salidas_lag_24",
    "llegadas_lag_24",
    "salidas_lag_168",
    "llegadas_lag_168",
    "festivo",
    "hora_sin",
    "hora_cos",
    "dia_semana_sin",
    "dia_semana_cos",
    "mes_sin",
    "mes_cos",
    "fecha_objetivo",
]

COLUMNAS_NUMERICAS_FLOAT: list[str] = [
    "hora",
    "dia_semana",
    "mes",
    "fin_semana",
    "salidas_lag_1",
    "llegadas_lag_1",
    "salidas_lag_24",
    "llegadas_lag_24",
    "salidas_lag_168",
    "llegadas_lag_168",
    "festivo",
    "hora_sin",
    "hora_cos",
    "dia_semana_sin",
    "dia_semana_cos",
    "mes_sin",
    "mes_cos",
]

EJEMPLO_REGISTRO_1: dict = {
    "estacion_id": 173,
    "fecha_hora": "2026-09-01T05:00:00",
    "hora": 6.0,
    "dia_semana": 1.0,
    "mes": 9.0,
    "fin_semana": 0.0,
    "salidas_lag_1": 0.0,
    "llegadas_lag_1": 0.0,
    "salidas_lag_24": 3.0,
    "llegadas_lag_24": 2.0,
    "salidas_lag_168": 4.0,
    "llegadas_lag_168": 1.0,
    "festivo": 0.0,
    "hora_sin": 1.0,
    "hora_cos": 6.123233995736766e-17,
    "dia_semana_sin": 0.7818314824680298,
    "dia_semana_cos": 0.6234898018587336,
    "mes_sin": -0.8660254037844384,
    "mes_cos": -0.5000000000000004,
    "fecha_objetivo": "2026-09-01T06:00:00",
}

EJEMPLO_REGISTRO_2: dict = {
    "estacion_id": 303,
    "fecha_hora": "2026-09-01T05:00:00",
    "hora": 6.0,
    "dia_semana": 1.0,
    "mes": 9.0,
    "fin_semana": 0.0,
    "salidas_lag_1": 0.0,
    "llegadas_lag_1": 0.0,
    "salidas_lag_24": 1.0,
    "llegadas_lag_24": 5.0,
    "salidas_lag_168": 0.0,
    "llegadas_lag_168": 3.0,
    "festivo": 0.0,
    "hora_sin": 1.0,
    "hora_cos": 6.123233995736766e-17,
    "dia_semana_sin": 0.7818314824680298,
    "dia_semana_cos": 0.6234898018587336,
    "mes_sin": -0.8660254037844384,
    "mes_cos": -0.5000000000000004,
    "fecha_objetivo": "2026-09-01T06:00:00",
}


def _max_lote_desde_params() -> int:
    api_cfg = cargar_parametros().get("api", {})
    limite = int(api_cfg["max_lote"])
    if limite < 1:
        raise ValueError("api.max_lote en params.yaml debe ser >= 1.")
    return limite


class RegistroEntrada(BaseModel):
    """Registro de entrada con las 20 variables requeridas por el modelo champion."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EJEMPLO_REGISTRO_1]},
    )

    estacion_id: int = Field(..., ge=1, description="Identificador único de la estación MiBici.")
    fecha_hora: datetime = Field(..., description="Fecha y hora de corte actual (t).")
    hora: float = Field(..., ge=0.0, le=23.0, description="Hora del día objetivo (0 a 23).")
    dia_semana: float = Field(
        ..., ge=0.0, le=6.0, description="Día de la semana objetivo (0 = lunes, 6 = domingo)."
    )
    mes: float = Field(..., ge=1.0, le=12.0, description="Mes del año objetivo (1 a 12).")
    fin_semana: float = Field(
        ..., ge=0.0, le=1.0, description="Indicador de fin de semana (1.0 si sábado/domingo)."
    )
    salidas_lag_1: float = Field(
        ..., ge=0.0, description="Salidas registradas 1 hora antes de la hora objetivo."
    )
    llegadas_lag_1: float = Field(
        ..., ge=0.0, description="Llegadas registradas 1 hora antes de la hora objetivo."
    )
    salidas_lag_24: float | None = Field(
        default=None,
        ge=0.0,
        description="Salidas registradas 24 horas antes (admite null si se imputa).",
    )
    llegadas_lag_24: float | None = Field(
        default=None,
        ge=0.0,
        description="Llegadas registradas 24 horas antes (admite null si se imputa).",
    )
    salidas_lag_168: float | None = Field(
        default=None,
        ge=0.0,
        description="Salidas registradas 168 horas antes (admite null si se imputa).",
    )
    llegadas_lag_168: float | None = Field(
        default=None,
        ge=0.0,
        description="Llegadas registradas 168 horas antes (admite null si se imputa).",
    )
    festivo: float = Field(
        ..., ge=0.0, le=1.0, description="Indicador de día festivo oficial (1.0 si festivo)."
    )
    hora_sin: float = Field(
        ..., ge=-1.0, le=1.0, description="Componente seno de la codificación cíclica de la hora."
    )
    hora_cos: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Componente coseno de la codificación cíclica de la hora.",
    )
    dia_semana_sin: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Componente seno de la codificación cíclica del día de la semana.",
    )
    dia_semana_cos: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Componente coseno de la codificación cíclica del día de la semana.",
    )
    mes_sin: float = Field(
        ..., ge=-1.0, le=1.0, description="Componente seno de la codificación cíclica del mes."
    )
    mes_cos: float = Field(
        ..., ge=-1.0, le=1.0, description="Componente coseno de la codificación cíclica del mes."
    )
    fecha_objetivo: datetime = Field(
        ..., description="Fecha y hora objetivo para la cual se pronostican las salidas (t + 1 h)."
    )


class SolicitudPrediccion(BaseModel):
    """Lote de registros estación-hora para consultar el pronóstico de demanda."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "registros": [
                        EJEMPLO_REGISTRO_1,
                        EJEMPLO_REGISTRO_2,
                    ]
                }
            ]
        },
    )

    registros: list[RegistroEntrada] = Field(
        ...,
        min_length=1,
        max_length=_max_lote_desde_params(),
        description="Lista de registros estación-hora a predecir (al menos 1 registro).",
    )

    def a_dataframe(self) -> pd.DataFrame:
        """Convierte los registros validados en un DataFrame con el orden y tipos del modelo."""
        filas = [registro.model_dump() for registro in self.registros]
        df = pd.DataFrame(filas, columns=COLUMNAS_ENTRADA)

        df["estacion_id"] = df["estacion_id"].astype("int64")
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"])

        for col in COLUMNAS_NUMERICAS_FLOAT:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        return df[COLUMNAS_ENTRADA]


class RespuestaPrediccion(BaseModel):
    """Respuesta con las salidas pronosticadas para la siguiente hora."""

    salidas_predichas: list[float] = Field(
        ...,
        description="Salidas estimadas para cada registro en el mismo orden de entrada (>= 0).",
    )
    n_predicciones: int = Field(
        ...,
        ge=1,
        description="Número total de predicciones generadas.",
    )


class RespuestaSalud(BaseModel):
    """Estado de salud del servicio y disponibilidad del modelo en memoria."""

    estado: str = Field(..., description="Estado general del servicio ('ok' o 'error').")
    modelo_cargado: bool = Field(
        ..., description="Indica si el modelo MLflow está cargado y listo para inferencia."
    )
    detalle: str | None = Field(
        default=None,
        description="Mensaje adicional cuando ocurre un problema al cargar el modelo.",
    )
