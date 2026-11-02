"""Validación walk-forward (ventana creciente) y métricas para el modelado."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from mibici.config import cargar_parametros


class WalkForwardSplitter:
    """Divide los datos en ventanas crecientes por mes.

    En cada ventana se entrena con todo lo anterior al mes de prueba y se evalúa en ese mes.
    Es compatible con scikit-learn (split y get_n_splits), así que se puede pasar como
    cv= a GridSearchCV o RandomizedSearchCV.

    brecha_horas quita del entrenamiento las últimas horas antes del mes de prueba: con
    horizonte de 1 h, el objetivo de las 23:00 del mes anterior cae en el mes de prueba.
    """

    def __init__(self, meses_prueba, columna_fecha="fecha_hora", brecha_horas=0):
        self.meses_prueba = [pd.Period(mes, freq="M") for mes in meses_prueba]
        self.columna_fecha = columna_fecha
        self.brecha_horas = brecha_horas

    @classmethod
    def desde_parametros(cls, params=None):
        """Crea el splitter con los meses de prueba y el horizonte de params.yaml."""
        params = params or cargar_parametros()
        validacion = params["validacion"]

        meses = pd.period_range(
            validacion["mes_prueba_inicio"], validacion["mes_prueba_fin"], freq="M"
        )

        return cls(
            meses_prueba=[str(mes) for mes in meses],
            brecha_horas=params["features"]["horizonte_horas"],
        )

    def split(self, X, y=None, groups=None):
        """Genera (índices de entrenamiento, índices de prueba), posicionales."""
        fechas = pd.to_datetime(pd.Series(X[self.columna_fecha]).to_numpy())

        for mes in self.meses_prueba:
            inicio_prueba = mes.start_time
            fin_entrenamiento = inicio_prueba - pd.Timedelta(hours=self.brecha_horas)

            entrenamiento = np.flatnonzero(fechas < fin_entrenamiento)
            prueba = np.flatnonzero((fechas >= inicio_prueba) & (fechas < (mes + 1).start_time))

            if len(entrenamiento) == 0 or len(prueba) == 0:
                raise ValueError(f"La ventana con prueba en {mes} no tiene datos suficientes.")

            yield entrenamiento, prueba

    def get_n_splits(self, X=None, y=None, groups=None):
        """Número de ventanas (una por mes de prueba)."""
        return len(self.meses_prueba)


def mae(y_real, y_pred):
    """Error absoluto medio."""
    return float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_pred))))


def rmse(y_real, y_pred):
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((np.asarray(y_real) - np.asarray(y_pred)) ** 2)))


def wape(y_real, y_pred):
    """Error absoluto ponderado: suma de errores absolutos entre la demanda total.

    Si la demanda total es 0 la métrica no está definida y se regresa NaN.
    """
    total = np.sum(np.abs(np.asarray(y_real)))

    if total == 0:
        return float("nan")

    return float(np.sum(np.abs(np.asarray(y_real) - np.asarray(y_pred))) / total)


METRICAS = {"mae": mae, "rmse": rmse, "wape": wape}


def evaluar_walk_forward(modelo, X, y, splitter):
    """Evalúa el modelo en cada ventana del splitter.

    En cada ventana se ajusta una copia nueva del modelo solo con el entrenamiento, para
    que el Pipeline (escalado, codificación) se reajuste sin ver el mes de prueba.
    Regresa la tabla por ventana y un resumen con la media y desviación de cada métrica.
    """
    y = np.asarray(y)
    filas = []

    for ventana, (entrenamiento, prueba) in enumerate(splitter.split(X, y), start=1):
        ajustado = clone(modelo).fit(X.iloc[entrenamiento], y[entrenamiento])
        y_pred = ajustado.predict(X.iloc[prueba])

        fila = {
            "ventana": ventana,
            "mes_prueba": str(splitter.meses_prueba[ventana - 1]),
            "n_entrenamiento": len(entrenamiento),
            "n_prueba": len(prueba),
        }

        for nombre, metrica in METRICAS.items():
            fila[nombre] = metrica(y[prueba], y_pred)

        filas.append(fila)

    tabla = pd.DataFrame(filas)

    resumen = {}
    for nombre in METRICAS:
        resumen[f"{nombre}_media"] = float(tabla[nombre].mean())
        resumen[f"{nombre}_desv"] = float(tabla[nombre].std(ddof=0))

    return tabla, resumen
