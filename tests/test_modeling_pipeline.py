"""Pruebas del Pipeline y entrenamiento de MiBici."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mibici.config import cargar_parametros
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.train import entrenar_y_evaluar


def crear_datos_prueba():
    """Genera datos pequeños para comprobar el modelado."""

    params = cargar_parametros()

    fechas = pd.date_range("2026-03-01", "2026-05-20", freq="12h")

    n = len(fechas)

    datos = pd.DataFrame(
        {
            "estacion_id": np.resize([1, 2], n),
            "fecha_hora": fechas,
            "fecha_objetivo": fechas + pd.Timedelta(hours=1),
            "salidas_siguiente_hora": np.arange(n) % 4,
        }
    )

    for tipo in ["salidas", "llegadas"]:
        for lag in params["features"]["rezagos"]:
            datos[f"{tipo}_lag_{lag}"] = np.arange(n) % 9

    for columna in [
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
        "fin_semana",
        "festivo",
    ]:
        datos[columna] = 0.0

    return datos


def test_pipeline_entrena_y_predice():
    """Comprueba las transformaciones y predicciones."""

    datos = crear_datos_prueba()

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"]

    modelo = crear_pipeline(Ridge(alpha=1.0))

    modelo.fit(X.iloc[:100], y.iloc[:100])

    # Estación desconocida y valor histórico faltante
    prueba = X.iloc[100:105].copy()
    prueba.loc[prueba.index[0], "estacion_id"] = 999
    prueba.loc[prueba.index[0], "salidas_lag_1"] = np.nan

    predicciones = modelo.predict(prueba)

    assert len(predicciones) == 5
    assert np.isfinite(predicciones).all()

    # Las fechas no deben entrar al estimador
    preprocesamiento = modelo.named_steps["preprocesamiento"]
    assert len(preprocesamiento.get_feature_names_out()) == 15


def test_entrenamiento_walk_forward():
    """Comprueba la evaluación temporal con dos ventanas."""

    datos = crear_datos_prueba()

    params = deepcopy(cargar_parametros())
    params["validacion"]["mes_prueba_inicio"] = "2026-04"
    params["validacion"]["mes_prueba_fin"] = "2026-05"

    modelo, tabla, resumen = entrenar_y_evaluar(
        datos=datos,
        estimador=Ridge(alpha=1.0),
        params=params,
        registrar=False,
    )

    assert len(tabla) == 2
    assert tabla["mes_prueba"].tolist() == ["2026-04", "2026-05"]

    assert np.isfinite(resumen["mae_media"])
    assert np.isfinite(resumen["rmse_media"])
    assert np.isfinite(resumen["wape_media"])

    assert hasattr(modelo, "predict")
