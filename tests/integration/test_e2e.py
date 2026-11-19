"""Pruebas de integración: datos, features, predicción, métricas y MLflow."""

from copy import deepcopy

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mibici.config import cargar_parametros
from mibici.features import generar_dataset_v3
from mibici.modeling import predict, tracking
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.validation import WalkForwardSplitter, mae, rmse, wape


@pytest.fixture
def mlflow_integracion_temporal(tmp_path, monkeypatch):
    """Aísla MLflow y el Model Registry del proyecto."""

    uri_anterior = mlflow.get_tracking_uri()
    uri_temporal = f"sqlite:///{tmp_path / 'mlflow_integracion.db'}"

    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", uri_temporal)
    monkeypatch.setattr(predict, "MLFLOW_TRACKING_URI", uri_temporal)
    monkeypatch.setattr(
        tracking,
        "MLFLOW_ARTIFACTS_DIR",
        tmp_path / "mlruns",
    )

    yield uri_temporal

    mlflow.set_tracking_uri(uri_anterior)


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


def test_flujo_completo_con_mlflow_temporal(tmp_path, mlflow_integracion_temporal):
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

    # 6. Registrar la evaluación con el módulo real de tracking
    tabla = pd.DataFrame(
        [
            {
                "ventana": 1,
                "mes_prueba": "2026-04",
                "n_entrenamiento": len(entrenamiento),
                "n_prueba": len(prueba),
                **metricas,
            }
        ]
    )

    resumen = {f"{nombre}_media": float(valor) for nombre, valor in metricas.items()}

    hiperparametros = {
        "algoritmo": "Ridge",
        "alpha": 1.0,
        "fuente": "sintetica",
    }

    tracking.configurar_mlflow("prueba-integracion-e2e")

    run_id = tracking.registrar_evaluacion(
        nombre_modelo="ridge_integracion",
        tabla=tabla,
        resumen=resumen,
        hiperparametros=hiperparametros,
        modelo_final=modelo,
        X_ejemplo=X.iloc[entrenamiento].tail(5),
    )

    # 7. Registrar el modelo con alias champion
    version = tracking.registrar_modelo(
        run_id=run_id,
        hiperparametros=hiperparametros,
        metricas=resumen,
        nombre="mibici-integracion-e2e",
        alias="champion",
    )

    assert int(version.version) == 1

    # 8. Verificar las métricas del run en MLflow
    cliente = mlflow.MlflowClient(tracking_uri=mlflow_integracion_temporal)

    registro = cliente.get_run(run_id)

    assert registro.info.status == "FINISHED"

    for nombre, valor in resumen.items():
        assert registro.data.metrics[nombre] == pytest.approx(valor)

    # 9. Recuperar el modelo desde el Registry
    modelo_cargado = predict.cargar_modelo(
        nombre="mibici-integracion-e2e",
        alias="champion",
    )

    # 10. Predecir mediante el módulo real de inferencia
    predicciones_recuperadas = predict.predecir(
        X.iloc[prueba],
        modelo=modelo_cargado,
    )

    predicciones_esperadas = np.clip(
        modelo.predict(X.iloc[prueba]),
        0,
        None,
    )

    np.testing.assert_allclose(
        predicciones_recuperadas,
        predicciones_esperadas,
    )


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


def test_reproducibilidad_modelo_guardado(mlflow_integracion_temporal):
    """Verifica la reproducibilidad usando MLflow Model Registry."""

    params, X, y, entrenamiento, prueba = preparar_datos_integracion()

    # Entrenar el Pipeline real
    modelo = crear_pipeline(Ridge(alpha=1.0), params=params)
    modelo.fit(X.iloc[entrenamiento], y[entrenamiento])

    datos_prueba = X.iloc[prueba]

    predicciones_originales = np.clip(
        modelo.predict(datos_prueba),
        0,
        None,
    )

    metricas = {
        "mae": mae(y[prueba], modelo.predict(datos_prueba)),
        "rmse": rmse(y[prueba], modelo.predict(datos_prueba)),
        "wape": wape(y[prueba], modelo.predict(datos_prueba)),
    }

    resumen = {f"{nombre}_media": float(valor) for nombre, valor in metricas.items()}

    tabla = pd.DataFrame(
        [
            {
                "ventana": 1,
                "mes_prueba": "2026-04",
                "n_entrenamiento": len(entrenamiento),
                "n_prueba": len(prueba),
                **metricas,
            }
        ]
    )

    hiperparametros = {"algoritmo": "Ridge", "alpha": 1.0}

    tracking.configurar_mlflow("prueba-reproducibilidad-e2e")

    run_id = tracking.registrar_evaluacion(
        nombre_modelo="ridge_reproducibilidad",
        tabla=tabla,
        resumen=resumen,
        hiperparametros=hiperparametros,
        modelo_final=modelo,
        X_ejemplo=X.iloc[entrenamiento].tail(5),
    )

    tracking.registrar_modelo(
        run_id=run_id,
        hiperparametros=hiperparametros,
        metricas=resumen,
        nombre="mibici-reproducibilidad-e2e",
        alias="champion",
    )

    # Cargar desde el Registry, no desde joblib
    modelo_cargado = predict.cargar_modelo(
        nombre="mibici-reproducibilidad-e2e",
        alias="champion",
    )

    predicciones_recuperadas = predict.predecir(
        datos_prueba,
        modelo=modelo_cargado,
    )

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
