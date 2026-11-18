"""Pruebas de integración: datos, features, predicción, métricas y MLflow."""

from copy import deepcopy

from joblib import dump, load
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mibici.config import cargar_parametros
from mibici.features import generar_dataset_v3
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.validation import WalkForwardSplitter, mae, rmse, wape


def crear_viajes_sinteticos():
    """Genera viajes de dos estaciones durante marzo y abril."""
    horas = pd.date_range(
        "2026-03-24 00:00",
        "2026-04-05 23:00",
        freq="h",
    )

    registros = []

    for hora in horas:
        for estacion in [6, 12]:
            cantidad = 1 + (hora.hour + estacion) % 3

            for i in range(cantidad):
                registros.append(
                    {
                        "Origen_Id": estacion,
                        "Destino_Id": 12 if estacion == 6 else 6,
                        "Inicio_del_viaje": hora + pd.Timedelta(minutes=5 + i),
                        "Fin_del_viaje": hora + pd.Timedelta(minutes=25 + i),
                    }
                )

    return pd.DataFrame(registros)


def test_flujo_completo_con_mlflow_temporal(tmp_path):
    """Comprueba el flujo completo sin depender de datos reales."""

    # 1. Crear, guardar y cargar datos sintéticos
    viajes = crear_viajes_sinteticos()
    ruta = tmp_path / "viajes_sinteticos.parquet"

    viajes.to_parquet(ruta, index=False)
    viajes_cargados = pd.read_parquet(ruta)

    assert len(viajes_cargados) == len(viajes)

    # 2. Generar el dataset v3 con el preprocesamiento real
    params = deepcopy(cargar_parametros())

    params["validacion"]["mes_prueba_inicio"] = "2026-04"
    params["validacion"]["mes_prueba_fin"] = "2026-04"

    datos = generar_dataset_v3(
        viajes_cargados,
        horizonte_horas=1,
        rezagos=params["features"]["rezagos"],
        fin_periodo="2026-04-05 23:00:00",
        inicio_extension="2026-04-04 00:00:00",
        dias_festivos=[],
    )

    assert not datos.empty
    assert datos["salidas_siguiente_hora"].notna().all()
    assert (datos["fecha_objetivo"] == datos["fecha_hora"] + pd.Timedelta(hours=1)).all()

    # 3. Separar entrenamiento y prueba sin fuga temporal
    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    splitter = WalkForwardSplitter.desde_parametros(params)
    entrenamiento, prueba = next(splitter.split(X, y))

    assert len(entrenamiento) > 0
    assert len(prueba) > 0
    assert X.iloc[entrenamiento]["fecha_objetivo"].max() < X.iloc[prueba]["fecha_hora"].min()

    # 4. Entrenar el Pipeline real y generar predicciones
    modelo = crear_pipeline(Ridge(alpha=1.0), params=params)

    modelo.fit(X.iloc[entrenamiento], y[entrenamiento])
    predicciones = modelo.predict(X.iloc[prueba])

    assert len(predicciones) == len(prueba)
    assert np.isfinite(predicciones).all()

    # 5. Calcular las métricas reales de la prueba
    metricas = {
        "mae": mae(y[prueba], predicciones),
        "rmse": rmse(y[prueba], predicciones),
        "wape": wape(y[prueba], predicciones),
    }

    assert all(np.isfinite(valor) for valor in metricas.values())
    assert all(valor >= 0 for valor in metricas.values())

    # 6. Registrar los resultados en MLflow temporal
    uri_anterior = mlflow.get_tracking_uri()
    uri_temporal = f"sqlite:///{tmp_path / 'mlflow_integracion.db'}"

    try:
        mlflow.set_tracking_uri(uri_temporal)
        mlflow.set_experiment("prueba-integracion-e2e")

        with mlflow.start_run(run_name="flujo_sintetico") as run:
            mlflow.log_param("fuente", "sintetica")
            mlflow.log_param("registros", len(datos))
            mlflow.log_metrics(metricas)

        cliente = mlflow.MlflowClient(tracking_uri=uri_temporal)
        registro = cliente.get_run(run.info.run_id)

        assert registro.info.status == "FINISHED"
        assert registro.data.metrics["mae"] == pytest.approx(metricas["mae"])
        assert registro.data.metrics["rmse"] == pytest.approx(metricas["rmse"])
        assert registro.data.metrics["wape"] == pytest.approx(metricas["wape"])

    finally:
        mlflow.set_tracking_uri(uri_anterior)


def test_rechaza_viajes_con_fechas_nulas():
    """Comprueba que se rechacen entradas incompletas."""

    viajes = crear_viajes_sinteticos().head(5).copy()
    viajes.loc[viajes.index[0], "Inicio_del_viaje"] = pd.NaT

    with pytest.raises(ValueError, match="valores nulos"):
        generar_dataset_v3(viajes)


def preparar_datos_integracion():
    """Prepara datos sintéticos para entrenar y evaluar el Pipeline."""

    params = deepcopy(cargar_parametros())
    params["validacion"]["mes_prueba_inicio"] = "2026-04"
    params["validacion"]["mes_prueba_fin"] = "2026-04"

    viajes = crear_viajes_sinteticos()

    datos = generar_dataset_v3(
        viajes,
        horizonte_horas=1,
        rezagos=params["features"]["rezagos"],
        fin_periodo="2026-04-05 23:00:00",
        inicio_extension="2026-04-04 00:00:00",
        dias_festivos=[],
    )

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    splitter = WalkForwardSplitter.desde_parametros(params)
    entrenamiento, prueba = next(splitter.split(X, y))

    return params, X, y, entrenamiento, prueba


def test_reproducibilidad_modelo_guardado(tmp_path):
    """Comprueba que el modelo conserva sus predicciones al cargarlo."""

    params, X, y, entrenamiento, prueba = preparar_datos_integracion()

    # Entrenar el Pipeline completo
    modelo = crear_pipeline(Ridge(alpha=1.0), params=params)
    modelo.fit(X.iloc[entrenamiento], y[entrenamiento])

    # Predicciones antes de guardar
    predicciones_originales = modelo.predict(X.iloc[prueba])

    # Guardar el modelo en un directorio temporal
    ruta_modelo = tmp_path / "modelo_integracion.joblib"
    dump(modelo, ruta_modelo)

    assert ruta_modelo.exists()

    # Cargar nuevamente el modelo
    modelo_cargado = load(ruta_modelo)

    # Repetir las predicciones con los mismos datos
    predicciones_recuperadas = modelo_cargado.predict(X.iloc[prueba])

    # Los resultados deben permanecer iguales
    np.testing.assert_allclose(
        predicciones_originales,
        predicciones_recuperadas,
        rtol=1e-10,
        atol=1e-10,
    )

    assert np.isfinite(predicciones_recuperadas).all()


def test_pipeline_con_estaciones_desconocidas_y_valores_faltantes():
    """Comprueba predicciones ante datos nuevos e incompletos."""

    params, X, y, entrenamiento, prueba = preparar_datos_integracion()

    # Entrenar con las estaciones conocidas
    modelo = crear_pipeline(Ridge(alpha=1.0), params=params)
    modelo.fit(X.iloc[entrenamiento], y[entrenamiento])

    # Tomar una muestra de los datos de prueba
    datos_prueba = X.iloc[prueba].head(5).copy()

    # Simular una estación que no apareció en entrenamiento
    datos_prueba.loc[datos_prueba.index[0], "estacion_id"] = 9999

    # Simular valores históricos faltantes
    datos_prueba.loc[datos_prueba.index[0], "salidas_lag_1"] = np.nan

    datos_prueba.loc[datos_prueba.index[1], "llegadas_lag_24"] = np.nan

    # El Pipeline debe imputar y codificar correctamente
    predicciones = modelo.predict(datos_prueba)

    assert len(predicciones) == 5
    assert np.isfinite(predicciones).all()

    # Las fechas no deben utilizarse como características del modelo
    nombres = modelo.named_steps["preprocesamiento"].get_feature_names_out()

    assert not any("fecha" in nombre for nombre in nombres)
