
"""Pruebas unitarias de los modelos de referencia de MiBici."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mibici.modeling.baselines import HistoricalMeanBaseline, WeeklyLagBaseline


class TestWeeklyLagBaseline:
    """Pruebas de la línea base de persistencia semanal."""

    @pytest.fixture
    def training_data(self):
        """Datos de entrenamiento con dos estaciones."""
        X = pd.DataFrame(
            {
                "estacion_id": [101, 101, 202, 202],
                "salidas_lag_168": [5.0, 10.0, 15.0, 20.0],
            }
        )

        y = np.array([10.0, 20.0, 30.0, 50.0])

        return X, y

    def test_whenWeeklyLagExists_thenUsesWeeklyLag(self, training_data):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        X_prueba = pd.DataFrame(
            {
                "estacion_id": [101, 202],
                "salidas_lag_168": [12.0, 25.0],
            }
        )

        predicciones = modelo.predict(X_prueba)

        np.testing.assert_array_equal(predicciones, [12.0, 25.0])

    def test_whenWeeklyLagIsMissing_thenUsesStationMean(self, training_data):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        X_prueba = pd.DataFrame(
            {
                "estacion_id": [101, 202],
                "salidas_lag_168": [np.nan, np.nan],
            }
        )

        predicciones = modelo.predict(X_prueba)

        # Estación 101: (10 + 20) / 2 = 15
        # Estación 202: (30 + 50) / 2 = 40
        np.testing.assert_array_equal(predicciones, [15.0, 40.0])

    def test_whenStationIsUnknown_thenUsesGlobalMean(self, training_data):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        X_prueba = pd.DataFrame(
            {
                "estacion_id": [999],
                "salidas_lag_168": [np.nan],
            }
        )

        predicciones = modelo.predict(X_prueba)

        # Media global: (10 + 20 + 30 + 50) / 4 = 27.5
        np.testing.assert_array_equal(predicciones, [27.5])

    def test_whenUnknownStationHasWeeklyLag_thenUsesWeeklyLag(
        self, training_data
    ):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        X_prueba = pd.DataFrame(
            {
                "estacion_id": [999],
                "salidas_lag_168": [8.0],
            }
        )

        predicciones = modelo.predict(X_prueba)

        np.testing.assert_array_equal(predicciones, [8.0])

    def test_whenPredicting_thenDoesNotModifyTrainingMeans(
        self, training_data
    ):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        medias_originales = modelo.medias_estacion_.copy()
        media_global_original = modelo.media_global_

        X_prueba = pd.DataFrame(
            {
                "estacion_id": [101, 999],
                "salidas_lag_168": [np.nan, np.nan],
            }
        )

        modelo.predict(X_prueba)

        assert modelo.medias_estacion_ == medias_originales
        assert modelo.media_global_ == media_global_original

    @pytest.mark.parametrize(
        "columna_faltante",
        ["estacion_id", "salidas_lag_168"],
    )
    def test_whenRequiredColumnIsMissing_thenFitRaisesValueError(
        self, training_data, columna_faltante
    ):
        X, y = training_data
        X = X.drop(columns=[columna_faltante])

        with pytest.raises(ValueError, match=columna_faltante):
            WeeklyLagBaseline().fit(X, y)

    @pytest.mark.parametrize(
        "columna_faltante",
        ["estacion_id", "salidas_lag_168"],
    )
    def test_whenRequiredColumnIsMissing_thenPredictRaisesValueError(
        self, training_data, columna_faltante
    ):
        X, y = training_data
        modelo = WeeklyLagBaseline().fit(X, y)

        X_prueba = X.drop(columns=[columna_faltante])

        with pytest.raises(ValueError, match=columna_faltante):
            modelo.predict(X_prueba)

    def test_whenModelIsNotFitted_thenPredictRaisesValueError(self):
        X = pd.DataFrame(
            {
                "estacion_id": [101],
                "salidas_lag_168": [5.0],
            }
        )

        with pytest.raises(ValueError, match="ajustarse"):
            WeeklyLagBaseline().predict(X)

    def test_whenModelIsCloned_thenCloneIsIndependent(self, training_data):
        X, y = training_data

        modelo = WeeklyLagBaseline().fit(X, y)
        modelo_clonado = clone(modelo)

        assert isinstance(modelo_clonado, WeeklyLagBaseline)
        assert not hasattr(modelo_clonado, "media_global_")

        modelo_clonado.fit(X, y)

        assert modelo_clonado.media_global_ == modelo.media_global_
        assert modelo_clonado is not modelo

        
class TestHistoricalMeanBaseline:
    """Pruebas del promedio histórico por estación, día y hora."""

    @pytest.fixture
    def training_data(self):
        X = pd.DataFrame(
            {
                "estacion_id": [101, 101, 101, 101, 202],
                "dia_semana": [0, 0, 1, 1, 0],
                "hora": [8, 8, 8, 9, 8],
            }
        )
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        return X, y

    def test_whenCombinationExists_thenUsesHistoricalMean(
        self, training_data
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)

        prueba = pd.DataFrame(
            {
                "estacion_id": [101, 101, 202],
                "dia_semana": [0, 1, 0],
                "hora": [8, 9, 8],
            }
        )

        np.testing.assert_allclose(
            modelo.predict(prueba), [15.0, 40.0, 50.0]
        )

    def test_whenCombinationIsUnknown_thenUsesStationMean(
        self, training_data
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)

        prueba = pd.DataFrame(
            {
                "estacion_id": [101, 202],
                "dia_semana": [6, 6],
                "hora": [23, 23],
            }
        )

        # Estación 101: (10 + 20 + 30 + 40) / 4 = 25
        # Estación 202: 50
        np.testing.assert_allclose(
            modelo.predict(prueba), [25.0, 50.0]
        )

    def test_whenStationIsUnknown_thenUsesGlobalMean(
        self, training_data
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)

        prueba = pd.DataFrame(
            {
                "estacion_id": [999],
                "dia_semana": [0],
                "hora": [8],
            }
        )

        np.testing.assert_allclose(
            modelo.predict(prueba), [30.0]
        )

    def test_whenPredicting_thenDoesNotModifyTrainingMeans(
        self, training_data
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)

        medias_originales = modelo.medias_historicas_.copy()
        estaciones_originales = modelo.medias_estacion_.copy()

        modelo.predict(
            pd.DataFrame(
                {
                    "estacion_id": [101, 999],
                    "dia_semana": [0, 6],
                    "hora": [8, 23],
                }
            )
        )

        assert modelo.medias_historicas_ == medias_originales
        assert modelo.medias_estacion_ == estaciones_originales

    @pytest.mark.parametrize(
        "columna",
        ["estacion_id", "dia_semana", "hora"],
    )
    def test_whenRequiredColumnIsMissing_thenFitRaisesValueError(
        self, training_data, columna
    ):
        X, y = training_data

        with pytest.raises(ValueError, match=columna):
            HistoricalMeanBaseline().fit(
                X.drop(columns=[columna]), y
            )

    @pytest.mark.parametrize(
        "columna",
        ["estacion_id", "dia_semana", "hora"],
    )
    def test_whenRequiredColumnIsMissing_thenPredictRaisesValueError(
        self, training_data, columna
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)

        with pytest.raises(ValueError, match=columna):
            modelo.predict(X.drop(columns=[columna]))

    def test_whenModelIsNotFitted_thenPredictRaisesValueError(self):
        prueba = pd.DataFrame(
            {
                "estacion_id": [101],
                "dia_semana": [0],
                "hora": [8],
            }
        )

        with pytest.raises(ValueError, match="ajustarse"):
            HistoricalMeanBaseline().predict(prueba)

    def test_whenModelIsCloned_thenCloneIsIndependent(
        self, training_data
    ):
        X, y = training_data
        modelo = HistoricalMeanBaseline().fit(X, y)
        clonado = clone(modelo)

        assert isinstance(clonado, HistoricalMeanBaseline)
        assert not hasattr(clonado, "medias_historicas_")

        clonado.fit(X, y)

        assert clonado.medias_historicas_ == modelo.medias_historicas_
        assert clonado is not modelo

