
"""Ajuste de hiperparámetros con validación temporal para MiBici."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterSampler
from pathlib import Path
from loguru import logger

from mibici.config import cargar_parametros
from mibici.modeling.pipeline import crear_pipeline
from mibici.modeling.validation import WalkForwardSplitter, evaluar_walk_forward
from mibici.modeling.validation import METRICAS

from mibici.modeling.checkpoints import (
    cargar_checkpoint,
    guardar_checkpoint,
    identificar_ejecucion,
)


def seleccionar_alpha_ridge(X, y, params=None):
    """Selecciona alpha usando únicamente los datos de entrenamiento disponibles."""

    if params is None:
        params = cargar_parametros()

    if "fecha_hora" not in X.columns:
        raise ValueError("Falta fecha_hora para la validación temporal.")

    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma cantidad de registros.")

    fechas = pd.to_datetime(X["fecha_hora"])
    meses = sorted(fechas.dt.to_period("M").unique())

    if len(meses) < 3:
        raise ValueError("Se requieren al menos tres meses para el ajuste.")

    # Utilizar los dos últimos meses disponibles como validación interna.
    meses_validacion = [str(mes) for mes in meses[-2:]]

    splitter = WalkForwardSplitter(
        meses_prueba=meses_validacion,
        brecha_horas=params["features"]["horizonte_horas"],
    )

    candidatos = params["modelado"]["hiperparametros"]["ridge"]["alpha"]
    resultados = []

    for alpha in candidatos:
        modelo = crear_pipeline(Ridge(alpha=alpha), params=params)

        tabla, resumen = evaluar_walk_forward(
            modelo=modelo,
            X=X,
            y=np.asarray(y),
            splitter=splitter,
        )

        resultados.append(
            {
                "alpha": alpha,
                "mae_media": resumen["mae_media"],
                "rmse_media": resumen["rmse_media"],
                "wape_media": resumen["wape_media"],
            }
        )

    resultados = pd.DataFrame(resultados)

    if not np.isfinite(resultados["mae_media"]).all():
        raise ValueError("La búsqueda produjo valores MAE no válidos.")

    # MAE es el criterio principal de selección.
    mejor_indice = resultados["mae_media"].idxmin()
    mejor_alpha = resultados.loc[mejor_indice, "alpha"]

    return float(mejor_alpha), resultados


def evaluar_ridge_ajustado(datos: pd.DataFrame, params=None):
    """Evalúa Ridge con ajuste de alpha dentro de cada ventana temporal."""

    if params is None:
        params = cargar_parametros()

    columnas_requeridas = {"fecha_hora", "salidas_siguiente_hora"}

    faltantes = columnas_requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    splitter = WalkForwardSplitter.desde_parametros(params)

    resultados = []

    for ventana, (entrenamiento, prueba) in enumerate(
        splitter.split(X, y), start=1
    ):
        X_entrenamiento = X.iloc[entrenamiento]
        y_entrenamiento = y[entrenamiento]

        X_prueba = X.iloc[prueba]
        y_prueba = y[prueba]

        mejor_alpha, _ = seleccionar_alpha_ridge(
            X_entrenamiento,
            y_entrenamiento,
            params=params,
        )

        modelo = crear_pipeline(
            Ridge(alpha=mejor_alpha),
            params=params,
        )

        modelo = clone(modelo).fit(
            X_entrenamiento,
            y_entrenamiento,
        )

        predicciones = modelo.predict(X_prueba)

        fila = {
            "ventana": ventana,
            "mes_prueba": str(splitter.meses_prueba[ventana - 1]),
            "n_entrenamiento": len(entrenamiento),
            "n_prueba": len(prueba),
            "alpha": mejor_alpha,
        }

        for nombre, metrica in METRICAS.items():
            fila[nombre] = metrica(y_prueba, predicciones)

        resultados.append(fila)

    tabla = pd.DataFrame(resultados)

    resumen = {}

    for nombre in METRICAS:
        resumen[f"{nombre}_media"] = float(tabla[nombre].mean())
        resumen[f"{nombre}_desv"] = float(
            tabla[nombre].std(ddof=0)
        )

    return tabla, resumen



def seleccionar_hiperparametros_hgb(
    X,
    y,
    params=None,
    checkpoint_dir: Path | None = None,
):
    """Selecciona hiperparámetros HGB con recuperación opcional."""

    if params is None:
        params = cargar_parametros()

    if "fecha_hora" not in X.columns:
        raise ValueError("Falta fecha_hora para la validación temporal.")

    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma cantidad de registros.")

    y = np.asarray(y)

    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Poisson requiere objetivos finitos y no negativos.")

    fechas = pd.to_datetime(X["fecha_hora"])
    meses = sorted(fechas.dt.to_period("M").unique())

    if len(meses) < 3:
        raise ValueError("Se requieren al menos tres meses para el ajuste.")

    splitter = WalkForwardSplitter(
        meses_prueba=[str(mes) for mes in meses[-2:]],
        brecha_horas=params["features"]["horizonte_horas"],
    )

    configuracion = params["modelado"]
    espacio = configuracion["hiperparametros"]["hist_gradient_boosting"]

    candidatos = list(
        ParameterSampler(
            param_distributions=espacio,
            n_iter=configuracion["busqueda"]["n_iter"],
            random_state=configuracion["semilla"],
        )
    )

    ruta_ejecucion = None

    if checkpoint_dir is not None:
        identificador = identificar_ejecucion(X, y, params)
        ruta_ejecucion = Path(checkpoint_dir) / identificador

        logger.info(
            f"Checkpoints HGB: {ruta_ejecucion}"
        )

    resultados = []

    for indice, candidato in enumerate(candidatos, start=1):
        # Convertir valores de NumPy a tipos JSON compatibles.
        parametros = {
            clave: (
                valor.item()
                if isinstance(valor, np.generic)
                else valor
            )
            for clave, valor in candidato.items()
        }

        ruta_checkpoint = (
            ruta_ejecucion / f"candidato_{indice:02d}.json"
            if ruta_ejecucion is not None
            else None
        )

        checkpoint = (
            cargar_checkpoint(ruta_checkpoint)
            if ruta_checkpoint is not None
            else None
        )

        if checkpoint is not None:
            if checkpoint.get("parametros") != parametros:
                raise ValueError(
                    f"Checkpoint incompatible: {ruta_checkpoint}"
                )

            metricas = checkpoint["metricas"]

            if not np.isfinite(metricas["mae_media"]):
                raise ValueError(
                    f"Checkpoint con MAE inválido: {ruta_checkpoint}"
                )

            logger.info(
                f"HGB candidato {indice}/{len(candidatos)} "
                "recuperado del checkpoint."
            )

        else:
            logger.info(
                f"Entrenando HGB candidato "
                f"{indice}/{len(candidatos)}: {parametros}"
            )

            estimador = HistGradientBoostingRegressor(
                **parametros,
                random_state=configuracion["semilla"],
            )

            modelo = crear_pipeline(estimador, params=params)

            _, resumen = evaluar_walk_forward(
                modelo=modelo,
                X=X,
                y=y,
                splitter=splitter,
            )

            metricas = {
                nombre: (
                    float(valor)
                    if np.isfinite(valor)
                    else None
                )
                for nombre, valor in (
                    (clave, resumen[clave])
                    for clave in (
                        "mae_media",
                        "rmse_media",
                        "wape_media",
                    )
                )
            }

            if metricas["mae_media"] is None:
                raise ValueError(
                    "La búsqueda produjo valores MAE no válidos."
                )

            if ruta_checkpoint is not None:
                guardar_checkpoint(
                    ruta_checkpoint,
                    {
                        "parametros": parametros,
                        "metricas": metricas,
                    },
                )

            logger.info(
                f"HGB candidato {indice}/{len(candidatos)} "
                f"completado. MAE: {metricas['mae_media']:.6f}"
            )

        resultados.append(
            {
                **parametros,
                **metricas,
            }
        )

    tabla = pd.DataFrame(resultados)

    if not np.isfinite(tabla["mae_media"]).all():
        raise ValueError("La búsqueda produjo valores MAE no válidos.")

    mejor_indice = tabla["mae_media"].idxmin()

    mejores_parametros = {
        clave: tabla.loc[mejor_indice, clave]
        for clave in espacio
    }

    logger.success(
        f"Búsqueda HGB completada. "
        f"Mejor MAE: {tabla.loc[mejor_indice, 'mae_media']:.6f}"
    )

    return mejores_parametros, tabla



def evaluar_hgb_ajustado(
    datos: pd.DataFrame,
    params=None,
    checkpoint_dir: Path | None = None,
):
    """Evalúa HGB Poisson con validación anidada y recuperación opcional."""

    if params is None:
        params = cargar_parametros()

    columnas_requeridas = {"fecha_hora", "salidas_siguiente_hora"}
    faltantes = columnas_requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(faltantes)}"
        )

    X = datos.drop(columns=["salidas_siguiente_hora"])
    y = datos["salidas_siguiente_hora"].to_numpy()

    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Poisson requiere objetivos finitos y no negativos.")

    splitter = WalkForwardSplitter.desde_parametros(params)

    ruta_ejecucion = None

    if checkpoint_dir is not None:
        identificador = identificar_ejecucion(X, y, params)
        ruta_ejecucion = Path(checkpoint_dir) / identificador

        logger.info(f"Checkpoints de evaluación HGB: {ruta_ejecucion}")

    resultados = []

    for ventana, (entrenamiento, prueba) in enumerate(
        splitter.split(X, y), start=1
    ):
        mes_prueba = str(splitter.meses_prueba[ventana - 1])

        ruta_ventana = (
            ruta_ejecucion / f"ventana_{ventana:02d}"
            if ruta_ejecucion is not None
            else None
        )

        ruta_resultado = (
            ruta_ventana / "resultado.json"
            if ruta_ventana is not None
            else None
        )

        checkpoint = (
            cargar_checkpoint(ruta_resultado)
            if ruta_resultado is not None
            else None
        )

        if checkpoint is not None:
            fila = checkpoint.get("resultado")

            if (
                not isinstance(fila, dict)
                or fila.get("mes_prueba") != mes_prueba
                or fila.get("n_entrenamiento") != len(entrenamiento)
                or fila.get("n_prueba") != len(prueba)
                or any(
                    not np.isfinite(fila.get(nombre, float("nan")))
                    for nombre in METRICAS
                )
            ):
                raise ValueError(
                    f"Checkpoint de ventana incompatible: {ruta_resultado}"
                )

            logger.info(
                f"HGB ventana {ventana}/{len(splitter.meses_prueba)} "
                f"({mes_prueba}) recuperada."
            )

            resultados.append(fila)
            continue

        logger.info(
            f"Iniciando HGB ventana "
            f"{ventana}/{len(splitter.meses_prueba)}: {mes_prueba}"
        )

        X_entrenamiento = X.iloc[entrenamiento]
        y_entrenamiento = y[entrenamiento]

        X_prueba = X.iloc[prueba]
        y_prueba = y[prueba]

        argumentos = {
            "X": X_entrenamiento,
            "y": y_entrenamiento,
            "params": params,
        }

        if ruta_ventana is not None:
            argumentos["checkpoint_dir"] = ruta_ventana / "candidatos"

        mejores_parametros, _ = seleccionar_hiperparametros_hgb(
            **argumentos
        )

        estimador = HistGradientBoostingRegressor(
            **mejores_parametros,
            random_state=params["modelado"]["semilla"],
        )

        modelo = crear_pipeline(estimador, params=params)
        modelo.fit(X_entrenamiento, y_entrenamiento)

        predicciones = modelo.predict(X_prueba)

        fila = {
            "ventana": ventana,
            "mes_prueba": mes_prueba,
            "n_entrenamiento": len(entrenamiento),
            "n_prueba": len(prueba),
            **{
                clave: (
                    valor.item()
                    if isinstance(valor, np.generic)
                    else valor
                )
                for clave, valor in mejores_parametros.items()
            },
        }

        for nombre, metrica in METRICAS.items():
            fila[nombre] = float(metrica(y_prueba, predicciones))

        if ruta_resultado is not None:
            guardar_checkpoint(
                ruta_resultado,
                {"resultado": fila},
            )

        resultados.append(fila)

        logger.success(
            f"HGB ventana {ventana}/{len(splitter.meses_prueba)} "
            f"completada. MAE: {fila['mae']:.6f}"
        )

    tabla = pd.DataFrame(resultados)

    resumen = {}
    for nombre in METRICAS:
        resumen[f"{nombre}_media"] = float(tabla[nombre].mean())
        resumen[f"{nombre}_desv"] = float(tabla[nombre].std(ddof=0))

    return tabla, resumen



