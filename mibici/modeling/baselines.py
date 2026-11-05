
"""Modelos de referencia para la predicción de demanda de MiBici."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class WeeklyLagBaseline(RegressorMixin, BaseEstimator):
    """Predice usando la demanda de la misma hora de la semana anterior."""

    def fit(self, X, y):
        """Calcula valores de respaldo utilizando solo entrenamiento."""
        if "estacion_id" not in X.columns:
            raise ValueError("Falta la columna estacion_id.")

        if "salidas_lag_168" not in X.columns:
            raise ValueError("Falta la columna salidas_lag_168.")

        datos = pd.DataFrame(
            {
                "estacion_id": X["estacion_id"].to_numpy(),
                "objetivo": np.asarray(y),
            }
        )

        self.media_global_ = float(datos["objetivo"].mean())

        self.medias_estacion_ = (
            datos.groupby("estacion_id")["objetivo"]
            .mean()
            .to_dict()
        )

        return self

    def predict(self, X):
        """Genera predicciones con respaldo para historiales incompletos."""
        if not hasattr(self, "media_global_"):
            raise ValueError("El modelo debe ajustarse antes de predecir.")

        if "estacion_id" not in X.columns:
            raise ValueError("Falta la columna estacion_id.")

        if "salidas_lag_168" not in X.columns:
            raise ValueError("Falta la columna salidas_lag_168.")

        predicciones = X["salidas_lag_168"].copy()

        respaldo = X["estacion_id"].map(self.medias_estacion_)
        respaldo = respaldo.fillna(self.media_global_)

        predicciones = predicciones.fillna(respaldo)

        return predicciones.to_numpy(dtype=float)

    
class HistoricalMeanBaseline(RegressorMixin, BaseEstimator):
    """Predice la media histórica por estación, día de semana y hora."""

    def fit(self, X, y):
        """Calcula las medias exclusivamente con datos de entrenamiento."""
        columnas = ["estacion_id", "dia_semana", "hora"]

        for columna in columnas:
            if columna not in X.columns:
                raise ValueError(f"Falta la columna {columna}.")

        datos = X[columnas].copy()
        datos["objetivo"] = np.asarray(y)

        self.media_global_ = float(datos["objetivo"].mean())

        self.medias_estacion_ = (
            datos.groupby("estacion_id")["objetivo"].mean().to_dict()
        )

        self.medias_historicas_ = (
            datos.groupby(columnas)["objetivo"].mean().to_dict()
        )

        return self

    def predict(self, X):
        """Predice con medias históricas y respaldos de entrenamiento."""
        if not hasattr(self, "medias_historicas_"):
            raise ValueError("El modelo debe ajustarse antes de predecir.")

        columnas = ["estacion_id", "dia_semana", "hora"]

        for columna in columnas:
            if columna not in X.columns:
                raise ValueError(f"Falta la columna {columna}.")

        claves = pd.MultiIndex.from_frame(X[columnas])

        predicciones = pd.Series(
            claves.map(self.medias_historicas_),
            index=X.index,
            dtype=float,
        )

        respaldo = X["estacion_id"].map(self.medias_estacion_)
        respaldo = respaldo.fillna(self.media_global_)

        return predicciones.fillna(respaldo).to_numpy(dtype=float)

