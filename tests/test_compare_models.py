
"""Pruebas de la comparación y registro de modelos en MLflow."""

from pathlib import Path

import pandas as pd

from mibici.modeling.compare_models import comparar_modelos


def test_whenModelsAreCompared_thenRegistersFourEvaluations(
    monkeypatch, tmp_path
):
    """Registra cuatro modelos sin repetir entrenamientos reales."""
    datos = pd.DataFrame(
        {
            "fecha_hora": [pd.Timestamp("2026-04-01")],
            "estacion_id": [1],
            "salidas_siguiente_hora": [2.0],
        }
    )

    params = {
        "mlflow": {"experimento": "experimento_prueba"},
        "features": {
            "horizonte_horas": 1,
            "rezagos": [1, 24, 168],
        },
    }

    llamadas = []
    registros = []

    def resultado(mae):
        tabla = pd.DataFrame(
            {
                "mes_prueba": ["2026-04"],
                "n_entrenamiento": [100],
                "n_prueba": [20],
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

    def evaluar_baselines_simulado(datos, params):
        llamadas.append("baselines")

        tabla_semanal, resumen_semanal = resultado(0.9)
        tabla_historico, resumen_historico = resultado(0.8)

        return {
            "baseline_lag_168": {
                "tabla": tabla_semanal,
                "resumen": resumen_semanal,
            },
            "baseline_media_historica": {
                "tabla": tabla_historico,
                "resumen": resumen_historico,
            },
        }

    def evaluar_ridge_simulado(datos, params):
        llamadas.append("ridge")
        return resultado(0.7)

    def evaluar_hgb_simulado(datos, params, checkpoint_dir):
        llamadas.append("hgb")
        assert Path(checkpoint_dir) == tmp_path / "checkpoints_hgb"
        return resultado(0.6)

    def configurar_mlflow_simulado(experimento):
        llamadas.append("configurar_mlflow")
        assert experimento == "experimento_prueba"

    def registrar_simulado(
        nombre_modelo,
        tabla,
        resumen,
        hiperparametros,
    ):
        registros.append(
            {
                "nombre": nombre_modelo,
                "tabla": tabla,
                "resumen": resumen,
                "hiperparametros": hiperparametros,
            }
        )
        return f"run_{nombre_modelo}"

    monkeypatch.setattr(
        "mibici.modeling.compare_models.evaluar_baselines",
        evaluar_baselines_simulado,
    )
    monkeypatch.setattr(
        "mibici.modeling.compare_models.evaluar_ridge_ajustado",
        evaluar_ridge_simulado,
    )
    monkeypatch.setattr(
        "mibici.modeling.compare_models.evaluar_hgb_ajustado",
        evaluar_hgb_simulado,
    )
    monkeypatch.setattr(
        "mibici.modeling.compare_models.configurar_mlflow",
        configurar_mlflow_simulado,
    )
    monkeypatch.setattr(
        "mibici.modeling.compare_models.registrar_evaluacion",
        registrar_simulado,
    )

    comparacion = comparar_modelos(
        datos,
        params=params,
        checkpoint_dir=tmp_path / "checkpoints_hgb",
    )

    assert llamadas == [
        "baselines",
        "ridge",
        "hgb",
        "configurar_mlflow",
    ]

    assert len(registros) == 4

    assert set(registro["nombre"] for registro in registros) == {
        "baseline_lag_168",
        "baseline_media_historica",
        "ridge_ajustado",
        "hgb_poisson",
    }

    assert comparacion["modelo"].tolist() == [
        "hgb_poisson",
        "ridge_ajustado",
        "baseline_media_historica",
        "baseline_lag_168",
    ]

    assert comparacion["mae_media"].tolist() == [
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    for registro in registros:
        assert registro["hiperparametros"]["validacion"] == "walk_forward"
        assert registro["hiperparametros"]["horizonte_horas"] == 1
        assert registro["hiperparametros"]["rezagos"] == "1,24,168"
        assert len(registro["tabla"]) == 1
