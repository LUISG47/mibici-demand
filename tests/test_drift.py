
"""Pruebas de simulación, detección, evaluación y alertas de drift."""

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring.alerts import evaluar_alertas
from mibici.monitoring.detectors import (
    calcular_ks,
    calcular_psi,
    detectar_drift,
)
from mibici.monitoring.drift import (
    COLUMNAS_HISTORICAS,
    ESCENARIOS,
    simular_drift,
)
from mibici.monitoring.evaluation import (
    calcular_metricas,
    evaluar_escenarios,
)
from mibici.monitoring.plots import generar_graficas


@pytest.fixture
def datos():
    """Muestra reproducible con las características del modelo."""
    rng = np.random.default_rng(42)
    n = 200

    datos = pd.DataFrame({
        columna: rng.integers(1, 100, size=n).astype(float)
        for columna in COLUMNAS_HISTORICAS
    })

    datos["fin_semana"] = np.tile([0, 1], n // 2)
    datos["festivo"] = 0
    datos["salidas_siguiente_hora"] = rng.integers(0, 20, size=n)
    datos["estacion_id"] = 1
    datos["fecha_hora"] = pd.date_range(
        "2026-09-01", periods=n, freq="h"
    )

    return datos


@pytest.fixture
def params_drift():
    return {
        "drift": {
            "semilla": 42,
            "simulacion": {
                "desplazamiento_medias": {"factor": 1.25},
                "features_faltantes": {"proporcion": 0.10},
                "cambio_estacional": {"factor": 1.20},
            },
            "umbrales": {
                "psi_advertencia": 0.10,
                "psi_critico": 0.25,
                "ks_pvalor": 0.05,
                "deterioro_mae_relativo": 0.10,
            },
        }
    }


class ModeloPrueba:
    """Modelo determinista para probar la evaluación sin entrenar."""

    def predict(self, X):
        return X["salidas_lag_1"].fillna(0).to_numpy(dtype=float) * 0.1


def test_when_simulatingDrift_then_originalDataRemainsUnchanged(
    datos, params_drift
):
    original = datos.copy(deep=True)

    for escenario in ESCENARIOS:
        simular_drift(datos, escenario, params=params_drift)

    pd.testing.assert_frame_equal(datos, original)


@pytest.mark.parametrize("escenario", ESCENARIOS)
def test_when_simulatingDrift_then_targetAndIdentifiersRemainUnchanged(
    datos, params_drift, escenario
):
    alterados = simular_drift(datos, escenario, params=params_drift)

    for columna in (
        "salidas_siguiente_hora",
        "estacion_id",
        "fecha_hora",
    ):
        pd.testing.assert_series_equal(
            alterados[columna],
            datos[columna],
        )


def test_when_shiftingMeans_then_historicalFeaturesAreScaled(
    datos, params_drift
):
    alterados = simular_drift(
        datos, "desplazamiento_medias", params=params_drift
    )

    esperado = datos[list(COLUMNAS_HISTORICAS)] * 1.25

    pd.testing.assert_frame_equal(
        alterados[list(COLUMNAS_HISTORICAS)],
        esperado,
    )


def test_when_addingMissingFeatures_then_resultIsReproducible(
    datos, params_drift
):
    primero = simular_drift(
        datos, "features_faltantes", params=params_drift
    )
    segundo = simular_drift(
        datos, "features_faltantes", params=params_drift
    )

    pd.testing.assert_frame_equal(primero, segundo)

    assert primero[list(COLUMNAS_HISTORICAS)].isna().sum().sum() > 0


def test_when_changingSeasonality_then_onlySpecialDaysAreModified(
    datos, params_drift
):
    alterados = simular_drift(
        datos, "cambio_estacional", params=params_drift
    )

    especiales = datos["fin_semana"].eq(1) | datos["festivo"].eq(1)
    columnas = list(COLUMNAS_HISTORICAS)

    pd.testing.assert_frame_equal(
        alterados.loc[~especiales, columnas],
        datos.loc[~especiales, columnas],
    )

    pd.testing.assert_frame_equal(
        alterados.loc[especiales, columnas],
        datos.loc[especiales, columnas] * 1.20,
    )


@pytest.mark.parametrize(
    "escenario",
    ["desconocido", "", "incremento"],
)
def test_when_scenarioIsUnsupported_then_raisesValueError(
    datos, params_drift, escenario
):
    with pytest.raises(ValueError, match="Escenario no soportado"):
        simular_drift(datos, escenario, params=params_drift)


def test_when_requiredColumnIsMissing_then_raisesValueError(
    datos, params_drift
):
    incompletos = datos.drop(columns=["salidas_lag_1"])

    with pytest.raises(ValueError, match="Faltan columnas"):
        simular_drift(
            incompletos,
            "desplazamiento_medias",
            params=params_drift,
        )


@pytest.mark.parametrize(
    ("escenario", "clave", "valor"),
    [
        ("desplazamiento_medias", "factor", 0),
        ("desplazamiento_medias", "factor", -1),
        ("features_faltantes", "proporcion", -0.1),
        ("features_faltantes", "proporcion", 1.1),
        ("cambio_estacional", "factor", 0),
    ],
)
def test_whenSimulationParametersAreInvalid_then_raisesValueError(
    datos, params_drift, escenario, clave, valor
):
    params_drift["drift"]["simulacion"][escenario][clave] = valor

    with pytest.raises(ValueError):
        simular_drift(datos, escenario, params=params_drift)


def test_whenDistributionsAreIdentical_then_psiAndKsAreZero():
    referencia = pd.Series(np.arange(100, dtype=float))

    assert calcular_psi(referencia, referencia) == pytest.approx(0)

    estadistico, pvalor = calcular_ks(referencia, referencia)

    assert estadistico == pytest.approx(0)
    assert pvalor == pytest.approx(1)


def test_whenDistributionShifts_then_psiAndKsDetectDifference():
    referencia = pd.Series(np.arange(1000, dtype=float))
    alterados = referencia + 500

    assert calcular_psi(referencia, alterados) > 0

    estadistico, pvalor = calcular_ks(referencia, alterados)

    assert estadistico > 0
    assert pvalor < 0.05


def test_whenMissingValuesIncrease_then_psiDetectsChange():
    referencia = pd.Series(np.arange(100, dtype=float))
    alterados = referencia.copy()
    alterados.iloc[:20] = np.nan

    assert calcular_psi(referencia, alterados) > 0


def test_whenReferenceIsEmpty_then_psiRaisesValueError():
    with pytest.raises(ValueError):
        calcular_psi(pd.Series(dtype=float), pd.Series([1.0]))


def test_whenKsHasNoValidValues_then_raisesValueError():
    with pytest.raises(ValueError):
        calcular_ks(pd.Series([np.nan]), pd.Series([1.0]))


def test_whenDetectingDrift_then_returnsMetricsForEachVariable():
    referencia = pd.DataFrame({
        "a": np.arange(100, dtype=float),
        "b": np.arange(100, dtype=float),
    })
    actual = referencia.copy()
    actual["a"] += 50

    resultado = detectar_drift(referencia, actual, ["a", "b"])

    assert len(resultado) == 2
    assert set(resultado["variable"]) == {"a", "b"}
    assert resultado.loc[
        resultado["variable"] == "a", "psi"
    ].iloc[0] > 0


def test_whenEvaluatingScenarios_then_returnsFourResults(
    datos, params_drift
):
    resultado = evaluar_escenarios(
        datos, ModeloPrueba(), params=params_drift
    )

    assert len(resultado) == 4
    assert set(resultado["escenario"]) == {
        "referencia", *ESCENARIOS
    }
    assert (resultado["n_registros"] == len(datos)).all()
    assert resultado.iloc[0]["deterioro_mae_relativo"] == 0


def test_whenTargetIsMissing_then_evaluationRaisesValueError(datos):
    with pytest.raises(ValueError, match="variable objetivo"):
        calcular_metricas(
            datos.drop(columns=["salidas_siguiente_hora"]),
            ModeloPrueba(),
        )


def test_whenDataIsEmpty_then_evaluationRaisesValueError(datos):
    with pytest.raises(ValueError, match="vacío"):
        calcular_metricas(datos.iloc[:0], ModeloPrueba())


def test_whenPsiIsCritical_then_alertIsCritical(params_drift):
    deteccion = pd.DataFrame([{
        "variable": "salidas_lag_1",
        "psi": 0.30,
        "ks_pvalor": 0.01,
        "nulos_referencia": 0.0,
        "nulos_actual": 0.0,
    }])

    resultado = evaluar_alertas(
        deteccion, 0.0, params=params_drift
    )

    assert resultado["nivel"] == "critico"
    assert resultado["drift_detectado"]
    assert resultado["variables_afectadas"] == ["salidas_lag_1"]


def test_whenOnlyPerformanceDegrades_then_alertIsWarning(params_drift):
    deteccion = pd.DataFrame([{
        "variable": "salidas_lag_1",
        "psi": 0.0,
        "ks_pvalor": 1.0,
        "nulos_referencia": 0.0,
        "nulos_actual": 0.0,
    }])

    resultado = evaluar_alertas(
        deteccion, 0.15, params=params_drift
    )

    assert resultado["nivel"] == "advertencia"
    assert resultado["deterioro_detectado"]
    assert not resultado["drift_detectado"]
    assert any("reentrenamiento" in accion for accion in resultado["acciones"])


def test_whenDriftAndDegradationOccur_then_alertIsCritical(params_drift):
    deteccion = pd.DataFrame([{
        "variable": "salidas_lag_1",
        "psi": 0.15,
        "ks_pvalor": 1.0,
        "nulos_referencia": 0.0,
        "nulos_actual": 0.0,
    }])

    resultado = evaluar_alertas(
        deteccion, 0.15, params=params_drift
    )

    assert resultado["nivel"] == "critico"
    assert resultado["drift_detectado"]
    assert resultado["deterioro_detectado"]


def test_whenGeneratingPlots_then_threePngFilesAreCreated(
    tmp_path, params_drift
):
    deteccion = pd.DataFrame([{
        "escenario": "desplazamiento_medias",
        "variable": "salidas_lag_1",
        "psi": 0.30,
        "ks_estadistico": 0.20,
    }])

    desempeno = pd.DataFrame([
        {"escenario": "referencia", "mae": 1.0},
        {"escenario": "desplazamiento_medias", "mae": 1.2},
    ])

    rutas = generar_graficas(
        deteccion,
        desempeno,
        directorio=tmp_path,
        params=params_drift,
    )

    assert set(rutas) == {"psi", "ks_estadistico", "mae"}
    assert all(ruta.is_file() for ruta in rutas.values())
    assert all(ruta.stat().st_size > 0 for ruta in rutas.values())
