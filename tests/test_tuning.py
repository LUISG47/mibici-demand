
"""Pruebas del ajuste de hiperparámetros de MiBici."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from mibici.config import cargar_parametros
from mibici.modeling.tuning import (
    evaluar_hgb_ajustado,
    evaluar_ridge_ajustado,
    seleccionar_alpha_ridge,
    seleccionar_hiperparametros_hgb,
)

@pytest.fixture
def params_prueba():
    """Configuración reducida para las pruebas."""
    params = deepcopy(cargar_parametros())
    params["modelado"]["hiperparametros"]["ridge"]["alpha"] = [
        0.1, 1.0, 10.0, 100.0
    ]
    return params


@pytest.fixture
def datos_prueba(params_prueba):
    """Genera cuatro meses de datos sintéticos para validación temporal."""
    fechas = pd.date_range(
        "2026-01-01",
        "2026-04-30",
        freq="12h",
    )

    n = len(fechas)

    X = pd.DataFrame(
        {
            "estacion_id": np.resize([101, 202], n),
            "fecha_hora": fechas,
        }
    )

    for tipo in ["salidas", "llegadas"]:
        for lag in params_prueba["features"]["rezagos"]:
            X[f"{tipo}_lag_{lag}"] = np.arange(n) % 11

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
        X[columna] = 0.0

    y = np.asarray(np.arange(n) % 7, dtype=float)

    return X, y


def test_whenCandidatesAreEvaluated_thenSelectsLowestMae(
    monkeypatch, datos_prueba, params_prueba
):
    """Selecciona el candidato con el menor MAE promedio."""
    X, y = datos_prueba
    alphas_evaluados = []

    def evaluar_simulado(modelo, X, y, splitter):
        alpha = modelo.named_steps["modelo"].alpha
        alphas_evaluados.append(alpha)

        assert [str(mes) for mes in splitter.meses_prueba] == [
            "2026-03",
            "2026-04",
        ]

        resumen = {
            "mae_media": abs(alpha - 10.0),
            "rmse_media": abs(alpha - 10.0) + 1,
            "wape_media": abs(alpha - 10.0) / 100,
        }

        return pd.DataFrame(), resumen

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    mejor_alpha, resultados = seleccionar_alpha_ridge(
        X, y, params=params_prueba
    )

    assert mejor_alpha == 10.0
    assert alphas_evaluados == [0.1, 1.0, 10.0, 100.0]
    assert len(resultados) == 4
    assert resultados.loc[
        resultados["mae_media"].idxmin(), "alpha"
    ] == 10.0


def test_whenUsingRealPipeline_thenEvaluatesAllCandidates(
    datos_prueba, params_prueba
):
    """Ejercita Ridge y las dos ventanas internas reales."""
    X, y = datos_prueba

    mejor_alpha, resultados = seleccionar_alpha_ridge(
        X, y, params=params_prueba
    )

    assert mejor_alpha in [0.1, 1.0, 10.0, 100.0]
    assert len(resultados) == 4

    assert set(resultados.columns) == {
        "alpha",
        "mae_media",
        "rmse_media",
        "wape_media",
    }

    assert np.isfinite(
        resultados[["mae_media", "rmse_media", "wape_media"]]
        .to_numpy()
    ).all()


def test_whenDateColumnIsMissing_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba

    with pytest.raises(ValueError, match="fecha_hora"):
        seleccionar_alpha_ridge(
            X.drop(columns=["fecha_hora"]),
            y,
            params=params_prueba,
        )


def test_whenInputLengthsDiffer_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba

    with pytest.raises(ValueError, match="misma cantidad"):
        seleccionar_alpha_ridge(
            X,
            y[:-1],
            params=params_prueba,
        )


def test_whenHistoryIsInsufficient_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba

    mascara = X["fecha_hora"] < "2026-03-01"

    with pytest.raises(ValueError, match="al menos tres meses"):
        seleccionar_alpha_ridge(
            X.loc[mascara].copy(),
            y[mascara.to_numpy()],
            params=params_prueba,
        )


def test_whenCandidateMaeIsInvalid_thenRaisesValueError(
    monkeypatch, datos_prueba, params_prueba
):
    X, y = datos_prueba

    def evaluar_simulado(modelo, X, y, splitter):
        return pd.DataFrame(), {
            "mae_media": float("nan"),
            "rmse_media": 1.0,
            "wape_media": 0.5,
        }

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    with pytest.raises(ValueError, match="MAE no válidos"):
        seleccionar_alpha_ridge(X, y, params=params_prueba)





def test_whenEvaluatingRidge_thenTuningUsesOnlyPastTrainingData(
    monkeypatch, datos_prueba, params_prueba
):
    """Comprueba que la búsqueda no utiliza el mes de prueba externo."""
    X, y = datos_prueba

    params_prueba["validacion"]["mes_prueba_inicio"] = "2026-03"
    params_prueba["validacion"]["mes_prueba_fin"] = "2026-04"

    datos = X.copy()
    datos["salidas_siguiente_hora"] = y

    meses_evaluados = []

    def seleccionar_simulado(X, y, params):
        mes_prueba = ["2026-03", "2026-04"][
            len(meses_evaluados)
        ]

        inicio_prueba = pd.Timestamp(mes_prueba)

        # Ningún registro del mes de prueba puede entrar al ajuste.
        assert (X["fecha_hora"] < inicio_prueba).all()

        # Respetar también la brecha de una hora.
        limite = inicio_prueba - pd.Timedelta(hours=1)
        assert (X["fecha_hora"] < limite).all()

        assert len(X) == len(y)

        meses_evaluados.append(mes_prueba)

        return 1.0, pd.DataFrame(
            {"alpha": [1.0], "mae_media": [0.5]}
        )

    monkeypatch.setattr(
        "mibici.modeling.tuning.seleccionar_alpha_ridge",
        seleccionar_simulado,
    )

    tabla, resumen = evaluar_ridge_ajustado(
        datos,
        params=params_prueba,
    )

    assert meses_evaluados == ["2026-03", "2026-04"]

    assert len(tabla) == 2
    assert tabla["mes_prueba"].tolist() == [
        "2026-03",
        "2026-04",
    ]

    assert tabla["alpha"].tolist() == [1.0, 1.0]

    for metrica in ["mae", "rmse", "wape"]:
        assert np.isfinite(tabla[metrica]).all()
        assert np.isfinite(resumen[f"{metrica}_media"])



def test_whenHgbCandidatesAreEvaluated_thenSelectsLowestMae(
    monkeypatch, datos_prueba, params_prueba
):
    """Selecciona el candidato con menor MAE."""
    X, y = datos_prueba

    params_prueba["modelado"]["busqueda"]["n_iter"] = 8

    candidatos_evaluados = []

    def evaluar_simulado(modelo, X, y, splitter):
        estimador = modelo.named_steps["modelo"]
        candidatos_evaluados.append(estimador.get_params())

        assert estimador.loss == "poisson"
        assert [str(mes) for mes in splitter.meses_prueba] == [
            "2026-03",
            "2026-04",
        ]
        assert splitter.brecha_horas == 1

        # Criterio controlado para identificar al mejor candidato.
        mae = float(estimador.learning_rate)

        return pd.DataFrame(), {
            "mae_media": mae,
            "rmse_media": mae + 1.0,
            "wape_media": mae / 2.0,
        }

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    mejores, resultados = seleccionar_hiperparametros_hgb(
        X, y, params=params_prueba
    )

    assert len(candidatos_evaluados) == 8
    assert len(resultados) == 8
    assert mejores["loss"] == "poisson"
    assert mejores["learning_rate"] == resultados["learning_rate"].min()
    assert resultados["mae_media"].notna().all()


def test_whenHgbSearchIsRepeated_thenCandidatesAreReproducible(
    monkeypatch, datos_prueba, params_prueba
):
    """La misma semilla produce la misma selección de candidatos."""
    X, y = datos_prueba

    def evaluar_simulado(modelo, X, y, splitter):
        estimador = modelo.named_steps["modelo"]

        return pd.DataFrame(), {
            "mae_media": float(estimador.learning_rate),
            "rmse_media": 1.0,
            "wape_media": 0.5,
        }

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    mejores_1, resultados_1 = seleccionar_hiperparametros_hgb(
        X, y, params=params_prueba
    )

    mejores_2, resultados_2 = seleccionar_hiperparametros_hgb(
        X, y, params=params_prueba
    )

    assert mejores_1 == mejores_2

    pd.testing.assert_frame_equal(
        resultados_1,
        resultados_2,
    )


@pytest.mark.parametrize(
    "valor_invalido",
    [-1.0, np.nan, np.inf],
)
def test_whenPoissonTargetIsInvalid_thenRaisesValueError(
    datos_prueba, params_prueba, valor_invalido
):
    """Poisson rechaza objetivos negativos o no finitos."""
    X, y = datos_prueba
    y = y.copy()
    y[0] = valor_invalido

    with pytest.raises(ValueError, match="Poisson"):
        seleccionar_hiperparametros_hgb(
            X, y, params=params_prueba
        )


def test_whenHgbDateColumnIsMissing_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba

    with pytest.raises(ValueError, match="fecha_hora"):
        seleccionar_hiperparametros_hgb(
            X.drop(columns=["fecha_hora"]),
            y,
            params=params_prueba,
        )


def test_whenHgbInputLengthsDiffer_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba

    with pytest.raises(ValueError, match="misma cantidad"):
        seleccionar_hiperparametros_hgb(
            X,
            y[:-1],
            params=params_prueba,
        )


def test_whenHgbHistoryIsInsufficient_thenRaisesValueError(
    datos_prueba, params_prueba
):
    X, y = datos_prueba
    mascara = X["fecha_hora"] < "2026-03-01"

    with pytest.raises(ValueError, match="al menos tres meses"):
        seleccionar_hiperparametros_hgb(
            X.loc[mascara].copy(),
            y[mascara.to_numpy()],
            params=params_prueba,
        )


def test_whenHgbCandidateMaeIsInvalid_thenRaisesValueError(
    monkeypatch, datos_prueba, params_prueba
):
    X, y = datos_prueba

    def evaluar_simulado(modelo, X, y, splitter):
        return pd.DataFrame(), {
            "mae_media": float("nan"),
            "rmse_media": 1.0,
            "wape_media": 0.5,
        }

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    with pytest.raises(ValueError, match="MAE no válidos"):
        seleccionar_hiperparametros_hgb(
            X, y, params=params_prueba
        )


def test_whenEvaluatingHgb_thenTuningUsesOnlyPastTrainingData(
    monkeypatch, datos_prueba, params_prueba
):
    """Comprueba la separación temporal en la evaluación HGB."""
    X, y = datos_prueba

    params_prueba["validacion"]["mes_prueba_inicio"] = "2026-03"
    params_prueba["validacion"]["mes_prueba_fin"] = "2026-04"

    datos = X.copy()
    datos["salidas_siguiente_hora"] = y

    meses_evaluados = []

    def seleccionar_simulado(X, y, params):
        mes_prueba = ["2026-03", "2026-04"][
            len(meses_evaluados)
        ]

        inicio_prueba = pd.Timestamp(mes_prueba)
        limite_entrenamiento = inicio_prueba - pd.Timedelta(
            hours=params["features"]["horizonte_horas"]
        )

        # La búsqueda no debe recibir datos del mes de prueba.
        assert (X["fecha_hora"] < limite_entrenamiento).all()
        assert len(X) == len(y)

        meses_evaluados.append(mes_prueba)

        return {
            "loss": "poisson",
            "learning_rate": 0.1,
            "max_iter": 20,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        }, pd.DataFrame()

    monkeypatch.setattr(
        "mibici.modeling.tuning.seleccionar_hiperparametros_hgb",
        seleccionar_simulado,
    )

    tabla, resumen = evaluar_hgb_ajustado(
        datos,
        params=params_prueba,
    )

    assert meses_evaluados == ["2026-03", "2026-04"]
    assert len(tabla) == 2
    assert tabla["mes_prueba"].tolist() == [
        "2026-03",
        "2026-04",
    ]

    assert (tabla["loss"] == "poisson").all()
    assert (tabla["learning_rate"] == 0.1).all()

    for metrica in ["mae", "rmse", "wape"]:
        assert np.isfinite(tabla[metrica]).all()
        assert np.isfinite(resumen[f"{metrica}_media"])



def test_whenHgbCheckpointsExist_thenSkipsCompletedCandidates(
    monkeypatch, datos_prueba, params_prueba, tmp_path
):
    """Recupera candidatos guardados sin repetir sus evaluaciones."""
    X, y = datos_prueba

    evaluaciones = []

    def evaluar_simulado(modelo, X, y, splitter):
        estimador = modelo.named_steps["modelo"]
        evaluaciones.append(estimador.get_params())

        mae = float(estimador.learning_rate)

        return pd.DataFrame(), {
            "mae_media": mae,
            "rmse_media": mae + 1.0,
            "wape_media": mae / 2.0,
        }

    monkeypatch.setattr(
        "mibici.modeling.tuning.evaluar_walk_forward",
        evaluar_simulado,
    )

    ruta = tmp_path / "checkpoints"

    # Primera ejecución: evalúa los ocho candidatos.
    mejores_1, tabla_1 = seleccionar_hiperparametros_hgb(
        X,
        y,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    assert len(evaluaciones) == 8
    assert len(list(ruta.rglob("candidato_*.json"))) == 8

    # Segunda ejecución: recupera todos los candidatos.
    mejores_2, tabla_2 = seleccionar_hiperparametros_hgb(
        X,
        y,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    # No debe realizar entrenamientos adicionales.
    assert len(evaluaciones) == 8

    assert mejores_1 == mejores_2

    pd.testing.assert_frame_equal(
        tabla_1,
        tabla_2,
    )


def test_whenHgbWindowsAreCompleted_thenResumesWithoutRetraining(
    monkeypatch, datos_prueba, params_prueba, tmp_path
):
    """Recupera ventanas completas sin repetir sus entrenamientos."""
    X, y = datos_prueba

    params_prueba["validacion"]["mes_prueba_inicio"] = "2026-03"
    params_prueba["validacion"]["mes_prueba_fin"] = "2026-04"

    datos = X.copy()
    datos["salidas_siguiente_hora"] = y

    llamadas_seleccion = []

    def seleccionar_simulado(X, y, params, checkpoint_dir=None):
        llamadas_seleccion.append(len(X))

        return {
            "loss": "poisson",
            "learning_rate": 0.1,
            "max_iter": 20,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        }, pd.DataFrame()

    monkeypatch.setattr(
        "mibici.modeling.tuning.seleccionar_hiperparametros_hgb",
        seleccionar_simulado,
    )

    ruta = tmp_path / "checkpoints_hgb"

    # Primera ejecución: entrena y guarda las dos ventanas.
    tabla_1, resumen_1 = evaluar_hgb_ajustado(
        datos,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    assert len(tabla_1) == 2
    assert len(llamadas_seleccion) == 2
    assert len(list(ruta.rglob("resultado.json"))) == 2

    # Segunda ejecución: recupera las ventanas guardadas.
    tabla_2, resumen_2 = evaluar_hgb_ajustado(
        datos,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    # No debe repetir la selección ni los entrenamientos.
    assert len(llamadas_seleccion) == 2

    pd.testing.assert_frame_equal(tabla_1, tabla_2)
    assert resumen_1 == resumen_2


def test_whenHgbExecutionIsPartiallyInterrupted_thenResumesPendingWindow(
    monkeypatch, datos_prueba, params_prueba, tmp_path
):
    """Recupera ventanas completas y ejecuta solo las pendientes."""
    X, y = datos_prueba

    params_prueba["validacion"]["mes_prueba_inicio"] = "2026-03"
    params_prueba["validacion"]["mes_prueba_fin"] = "2026-04"

    datos = X.copy()
    datos["salidas_siguiente_hora"] = y

    llamadas_seleccion = []

    def seleccionar_simulado(X, y, params, checkpoint_dir=None):
        llamadas_seleccion.append(len(X))

        return {
            "loss": "poisson",
            "learning_rate": 0.1,
            "max_iter": 20,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        }, pd.DataFrame()

    monkeypatch.setattr(
        "mibici.modeling.tuning.seleccionar_hiperparametros_hgb",
        seleccionar_simulado,
    )

    ruta = tmp_path / "checkpoints_hgb"

    tabla_original, resumen_original = evaluar_hgb_ajustado(
        datos,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    assert len(llamadas_seleccion) == 2

    checkpoints = sorted(ruta.rglob("resultado.json"))
    assert len(checkpoints) == 2

    # Simular pérdida del resultado de la segunda ventana.
    checkpoints[1].unlink()

    tabla_recuperada, resumen_recuperado = evaluar_hgb_ajustado(
        datos,
        params=params_prueba,
        checkpoint_dir=ruta,
    )

    # Solo debe ejecutarse nuevamente la ventana pendiente.
    assert len(llamadas_seleccion) == 3

    assert len(list(ruta.rglob("resultado.json"))) == 2

    pd.testing.assert_frame_equal(
        tabla_original,
        tabla_recuperada,
    )

    assert resumen_original == resumen_recuperado





