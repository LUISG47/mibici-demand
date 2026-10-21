"""Pruebas del preprocesamiento de MiBici."""

import pandas as pd

from mibici.features import (
    crear_codificador_estaciones,
    generar_dataset_estacion_hora,
    normalizar_variables_historicas,
)


def test_agregacion_estacion_hora():
    """Verifica conteos, horas sin viajes y variables temporales."""

    viajes = pd.DataFrame(
        {
            "Origen_Id": [2, 2],
            "Destino_Id": [3, 3],
            "Inicio_del_viaje": pd.to_datetime(["2025-11-17 08:05:00", "2025-11-17 10:10:00"]),
            "Fin_del_viaje": pd.to_datetime(["2025-11-17 08:30:00", "2025-11-17 10:30:00"]),
        }
    )

    resultado = generar_dataset_estacion_hora(viajes)

    assert resultado["salidas"].sum() == 2
    assert resultado["llegadas"].sum() == 2

    estacion = resultado[resultado["estacion_id"] == 2]
    hora_cero = estacion[estacion["hora"] == 9].iloc[0]

    assert hora_cero["salidas"] == 0
    assert hora_cero["llegadas"] == 0
    assert hora_cero["salidas_lag_1"] == 1
    assert hora_cero["festivo"] == 1


def test_normalizacion_sin_fuga_temporal():
    """El escalador se ajusta solamente con entrenamiento."""

    fechas = pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-01"])

    columnas = [f"{tipo}_lag_{lag}" for tipo in ["salidas", "llegadas"] for lag in [1, 24, 168]]

    datos = pd.DataFrame({"fecha_hora": fechas})

    for columna in columnas:
        datos[columna] = [1.0, 3.0, 1000.0]

    resultado, escalador, medianas = normalizar_variables_historicas(datos)

    assert len(resultado) == 3
    assert len(medianas) == 6

    # La media del entrenamiento es 2, no incluye agosto
    assert all(media == 2.0 for media in escalador.mean_)

    columnas_norm = [columna for columna in resultado if columna.endswith("_norm")]

    assert not resultado[columnas_norm].isna().any().any()


def test_codificacion_estaciones():
    """Comprueba que estaciones nuevas no provoquen errores."""

    datos = pd.DataFrame(
        {
            "fecha_hora": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-08-01"]),
            "estacion_id": [2, 3, 999],
        }
    )

    codificador = crear_codificador_estaciones(datos)

    assert len(codificador.categories_[0]) == 2

    nueva_estacion = pd.DataFrame({"estacion_id": [999]})
    transformado = codificador.transform(nueva_estacion)

    assert transformado.shape == (1, 2)
    assert transformado.nnz == 0


def test_cierre_periodo_y_extension_horaria():
    """Verifica el cierre del periodo y las horas sin movimientos."""

    viajes = pd.DataFrame(
        {
            "Origen_Id": [1, 1, 2],
            "Destino_Id": [1, 1, 2],
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-09-30 21:10:00",
                    "2026-09-30 22:20:00",
                    "2026-09-28 08:10:00",
                ]
            ),
            "Fin_del_viaje": pd.to_datetime(
                [
                    "2026-09-30 21:30:00",
                    "2026-10-01 00:15:00",
                    "2026-09-28 08:30:00",
                ]
            ),
        }
    )

    resultado = generar_dataset_estacion_hora(viajes)

    # No deben existir registros posteriores a septiembre
    fin_periodo = pd.Timestamp("2026-09-30 23:00:00")
    assert resultado["fecha_hora"].max() == fin_periodo
    assert (resultado["fecha_hora"] <= fin_periodo).all()

    # Estación activa: extender hasta las 23:00
    estacion_activa = resultado[resultado["estacion_id"] == 1].set_index("fecha_hora")

    assert fin_periodo in estacion_activa.index
    assert estacion_activa.loc[fin_periodo, "salidas"] == 0
    assert estacion_activa.loc[fin_periodo, "llegadas"] == 0

    # Estación inactiva antes del 29: no extender
    estacion_inactiva = resultado[resultado["estacion_id"] == 2]

    assert estacion_inactiva["fecha_hora"].max() == pd.Timestamp("2026-09-28 08:00:00")

    # Las salidas se conservan y la llegada de octubre se excluye
    assert resultado["salidas"].sum() == 3
    assert resultado["llegadas"].sum() == 2
