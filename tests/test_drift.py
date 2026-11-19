"""Pruebas de simulación, detección, evaluación y alertas de drift."""

from unittest.mock import patch

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
from mibici.monitoring.tracking import registrar_drift


@pytest.fixture
def datos():
    """Muestra reproducible con las características del modelo."""
    rng = np.random.default_rng(42)
    n = 200

    datos = pd.DataFrame(
        {columna: rng.integers(1, 100, size=n).astype(float) for columna in COLUMNAS_HISTORICAS}
    )

    datos["fin_semana"] = np.tile([0, 1], n // 2)
    datos["festivo"] = 0
    datos["salidas_siguiente_hora"] = rng.integers(0, 20, size=n)
    datos["estacion_id"] = 1
    datos["fecha_hora"] = pd.date_range("2026-09-01", periods=n, freq="h")

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
                "estres_caracteristicas": {"factor": 1.50},
            },
            "umbrales": {
                "psi_advertencia": 0.10,
                "psi_critico": 0.25,
                "ks_estadistico": 0.10,
                "deterioro_mae_relativo": 0.10,
            },
        }
    }


class ModeloPrueba:
    """Modelo determinista para probar la evaluación sin entrenar."""

    def predict(self, X):
        return X["salidas_lag_1"].fillna(0).to_numpy(dtype=float) * 0.1


def test_when_simulatingDrift_then_originalDataRemainsUnchanged(datos, params_drift):
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


def test_when_shiftingMeans_then_historicalFeaturesAreScaled(datos, params_drift):
    alterados = simular_drift(datos, "desplazamiento_medias", params=params_drift)

    esperado = datos[list(COLUMNAS_HISTORICAS)] * 1.25

    pd.testing.assert_frame_equal(
        alterados[list(COLUMNAS_HISTORICAS)],
        esperado,
    )


def test_whenStressingFeatures_then_historicalFeaturesAreScaled(datos, params_drift):
    alterados = simular_drift(
        datos,
        "estres_caracteristicas",
        params=params_drift,
    )

    esperado = datos[list(COLUMNAS_HISTORICAS)] * 1.50

    pd.testing.assert_frame_equal(
        alterados[list(COLUMNAS_HISTORICAS)],
        esperado,
    )

    pd.testing.assert_series_equal(
        alterados["salidas_siguiente_hora"],
        datos["salidas_siguiente_hora"],
    )


def test_when_addingMissingFeatures_then_resultIsReproducible(datos, params_drift):
    primero = simular_drift(datos, "features_faltantes", params=params_drift)
    segundo = simular_drift(datos, "features_faltantes", params=params_drift)

    pd.testing.assert_frame_equal(primero, segundo)

    assert primero[list(COLUMNAS_HISTORICAS)].isna().sum().sum() > 0


def test_when_changingSeasonality_then_onlySpecialDaysAreModified(datos, params_drift):
    alterados = simular_drift(datos, "cambio_estacional", params=params_drift)

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
def test_when_scenarioIsUnsupported_then_raisesValueError(datos, params_drift, escenario):
    with pytest.raises(ValueError, match="Escenario no soportado"):
        simular_drift(datos, escenario, params=params_drift)


def test_when_requiredColumnIsMissing_then_raisesValueError(datos, params_drift):
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


def test_whenPsiIntervalsChange_then_detectionUsesConfiguredIntervals():
    referencia = pd.DataFrame(
        {
            "demanda": np.arange(1, 101, dtype=float),
        }
    )
    actual = pd.DataFrame(
        {
            "demanda": np.arange(31, 131, dtype=float),
        }
    )

    resultado_dos = detectar_drift(
        referencia,
        actual,
        ["demanda"],
        n_intervalos=2,
    )

    resultado_diez = detectar_drift(
        referencia,
        actual,
        ["demanda"],
        n_intervalos=10,
    )

    psi_dos = resultado_dos.iloc[0]["psi"]
    psi_diez = resultado_diez.iloc[0]["psi"]

    assert psi_dos == pytest.approx(
        calcular_psi(referencia["demanda"], actual["demanda"], n_intervalos=2)
    )
    assert psi_diez == pytest.approx(
        calcular_psi(referencia["demanda"], actual["demanda"], n_intervalos=10)
    )
    assert psi_dos != pytest.approx(psi_diez)


def test_whenMissingValuesIncrease_then_psiDetectsChange():
    referencia = pd.Series(np.arange(100, dtype=float))
    alterados = referencia.copy()
    alterados.iloc[:20] = np.nan

    assert calcular_psi(referencia, alterados) > 0


@pytest.mark.parametrize(
    ("ceros_actuales", "positivos_actuales"),
    [
        (800, 200),
        (300, 700),
    ],
)
def test_whenZeroProportionChanges_then_psiDetectsDrift(ceros_actuales, positivos_actuales):
    referencia = pd.Series(np.concatenate([np.zeros(600), np.arange(1, 401)]))
    actual = pd.Series(
        np.concatenate(
            [
                np.zeros(ceros_actuales),
                np.arange(1, positivos_actuales + 1),
            ]
        )
    )

    assert calcular_psi(referencia, actual) > 0


def test_whenZeroHeavyDistributionsAreIdentical_then_psiIsZero():
    referencia = pd.Series(np.concatenate([np.zeros(600), np.arange(1, 401)]))

    assert calcular_psi(referencia, referencia.copy()) == pytest.approx(0)


def test_whenReferenceContainsOnlyZeros_then_psiDetectsNewValues():
    referencia = pd.Series(np.zeros(1000))
    actual = pd.Series(np.concatenate([np.zeros(800), np.ones(200)]))

    assert calcular_psi(referencia, actual) > 0


def test_whenReferenceIsEmpty_then_psiRaisesValueError():
    with pytest.raises(ValueError):
        calcular_psi(pd.Series(dtype=float), pd.Series([1.0]))


def test_whenKsHasNoValidValues_then_raisesValueError():
    with pytest.raises(ValueError):
        calcular_ks(pd.Series([np.nan]), pd.Series([1.0]))


def test_whenDetectingDrift_then_returnsMetricsForEachVariable():
    referencia = pd.DataFrame(
        {
            "a": np.arange(100, dtype=float),
            "b": np.arange(100, dtype=float),
        }
    )
    actual = referencia.copy()
    actual["a"] += 50

    resultado = detectar_drift(referencia, actual, ["a", "b"])

    assert len(resultado) == 2
    assert set(resultado["variable"]) == {"a", "b"}
    assert resultado.loc[resultado["variable"] == "a", "psi"].iloc[0] > 0


def test_whenEvaluatingScenarios_then_returnsAllResults(datos, params_drift):
    resultado = evaluar_escenarios(datos, ModeloPrueba(), params=params_drift)

    assert len(resultado) == len(ESCENARIOS) + 1
    assert set(resultado["escenario"]) == {
        "referencia",
        *ESCENARIOS,
    }

    assert (resultado["n_registros"] == len(datos)).all()
    assert resultado["mae"].notna().all()
    assert resultado["rmse"].notna().all()
    assert resultado["wape"].notna().all()

    referencia = resultado.loc[resultado["escenario"] == "referencia"].iloc[0]

    assert referencia["deterioro_mae_relativo"] == pytest.approx(0)


def test_whenTargetIsMissing_then_evaluationRaisesValueError(datos):
    with pytest.raises(ValueError, match="variable objetivo"):
        calcular_metricas(
            datos.drop(columns=["salidas_siguiente_hora"]),
            ModeloPrueba(),
        )


def test_whenDataIsEmpty_then_evaluationRaisesValueError(datos):
    with pytest.raises(ValueError, match="vacío"):
        calcular_metricas(datos.iloc[:0], ModeloPrueba())


def test_whenPsiIsCriticalWithoutPerformanceDegradation_then_generalAlertIsWarning(
    params_drift,
):
    deteccion = pd.DataFrame(
        [
            {
                "variable": "salidas_lag_1",
                "psi": 0.30,
                "ks_estadistico": 0.0005,
                "nulos_referencia": 0.0,
                "nulos_actual": 0.10,
            }
        ]
    )

    resultado = evaluar_alertas(
        deteccion,
        0.0232,
        params=params_drift,
    )

    assert resultado["nivel"] == "advertencia"
    assert resultado["drift_detectado"]
    assert not resultado["deterioro_detectado"]
    assert resultado["variables_afectadas"] == ["salidas_lag_1"]

    detalle = resultado["detalle"].iloc[0]

    assert detalle["nivel"] == "critico"
    assert detalle["alerta_psi"]
    assert not detalle["alerta_ks"]
    assert detalle["alerta_nulos"]


def test_whenOnlyPerformanceDegrades_then_alertIsCritical(params_drift):
    deteccion = pd.DataFrame(
        [
            {
                "variable": "salidas_lag_1",
                "psi": 0.0,
                "ks_estadistico": 0.0,
                "nulos_referencia": 0.0,
                "nulos_actual": 0.0,
            }
        ]
    )

    resultado = evaluar_alertas(
        deteccion,
        0.15,
        params=params_drift,
    )

    assert resultado["nivel"] == "critico"
    assert resultado["deterioro_detectado"]
    assert not resultado["drift_detectado"]
    assert resultado["variables_afectadas"] == []
    assert any("reentrenamiento" in accion for accion in resultado["acciones"])


@pytest.mark.parametrize(
    ("ks_estadistico", "drift_esperado"),
    [
        (0.099, False),
        (0.100, True),
        (0.101, True),
    ],
)
def test_whenKsStatisticReachesThreshold_then_driftIsDetected(
    params_drift, ks_estadistico, drift_esperado
):
    deteccion = pd.DataFrame(
        [
            {
                "variable": "salidas_lag_1",
                "psi": 0.0,
                "ks_estadistico": ks_estadistico,
                "nulos_referencia": 0.0,
                "nulos_actual": 0.0,
            }
        ]
    )

    resultado = evaluar_alertas(
        deteccion,
        0.0,
        params=params_drift,
    )

    assert resultado["drift_detectado"] is drift_esperado
    assert resultado["detalle"].iloc[0]["alerta_ks"] == drift_esperado

    nivel_esperado = "advertencia" if drift_esperado else "sin_alerta"
    assert resultado["nivel"] == nivel_esperado


def test_whenGeneratingPlots_then_threePngFilesAreCreated(tmp_path, params_drift):
    deteccion = pd.DataFrame(
        [
            {
                "escenario": "desplazamiento_medias",
                "variable": "salidas_lag_1",
                "psi": 0.30,
                "ks_estadistico": 0.20,
            }
        ]
    )

    desempeno = pd.DataFrame(
        [
            {"escenario": "referencia", "mae": 1.0},
            {"escenario": "desplazamiento_medias", "mae": 1.2},
        ]
    )

    rutas = generar_graficas(
        deteccion,
        desempeno,
        directorio=tmp_path,
        params=params_drift,
    )

    assert set(rutas) == {"psi", "ks_estadistico", "mae"}
    assert all(ruta.is_file() for ruta in rutas.values())
    assert all(ruta.stat().st_size > 0 for ruta in rutas.values())


def test_whenExperimentIsConfigured_then_trackingUsesConfiguredName(
    params_drift,
):
    params_drift["drift"]["experimento_mlflow"] = "experimento-prueba"
    params_drift["drift"]["mes_referencia"] = "2026-09"
    params_drift["drift"]["mae_referencia"] = 0.801

    deteccion = pd.DataFrame(
        {
            "variable": ["salidas_lag_1"],
            "psi": [0.15],
            "ks_pvalor": [0.01],
            "nulos_referencia": [0.0],
            "nulos_actual": [0.0],
        }
    )

    metricas = {
        "mae": 0.9,
        "rmse": 1.5,
        "wape": 0.6,
        "deterioro_mae_relativo": 0.12,
    }

    alertas = {
        "nivel": "critico",
        "drift_detectado": True,
        "deterioro_detectado": True,
        "variables_afectadas": ["salidas_lag_1"],
        "acciones": ["Evaluar reentrenamiento."],
    }

    with (
        patch("mibici.monitoring.tracking.configurar_mlflow") as configurar,
        patch("mibici.monitoring.tracking.mlflow") as mlflow_mock,
    ):
        mlflow_mock.start_run.return_value.__enter__.return_value.info.run_id = "run-prueba"

        resultado = registrar_drift(
            escenario="estres_caracteristicas",
            deteccion=deteccion,
            metricas=metricas,
            alertas=alertas,
            params=params_drift,
        )

    configurar.assert_called_once_with("experimento-prueba")
    assert resultado == "run-prueba"
