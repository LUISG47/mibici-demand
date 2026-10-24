"""
Pruebas unitarias para mibici.eda.
"""

import pandas as pd

from mibici.eda import ViajesEDA


def fixture_viajes_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3, 4],
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-01-05 08:00:00",  # Lunes 8h
                    "2026-01-05 08:30:00",  # Lunes 8h
                    "2026-01-06 18:00:00",  # Martes 18h
                    "2026-01-11 12:00:00",  # Domingo 12h
                ]
            ),
            "Fin_del_viaje": pd.to_datetime(
                [
                    "2026-01-05 08:15:00",
                    "2026-01-05 08:45:00",
                    "2026-01-06 18:20:00",
                    "2026-01-11 12:30:00",
                ]
            ),
            "Origen_Id": [10, 10, 20, 30],
            "Destino_Id": [20, 20, 10, 10],
            "duracion_min": [15.0, 15.0, 20.0, 30.0],
            "duracion_atipica": [False, False, False, False],
            "estacion_en_catalogo": [True, True, True, True],
            "Año_de_nacimiento": [1990.0, 1985.0, 1995.0, pd.NA],
            "Genero": ["M", "F", "M", "F"],
        }
    )


def fixture_nomenclatura_prueba() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 20, 30],
            "name": ["Estación 10", "Estación 20", "Estación 30"],
            "dpcapacity": [20, 15, 10],
            "latitude": [20.6, 20.7, 20.8],
            "longitude": [-103.3, -103.4, -103.5],
        }
    )


def test_demanda_por_hora():
    eda = ViajesEDA(fixture_viajes_prueba())
    demanda = eda.obtener_demanda_por_hora()
    assert len(demanda) == 24
    assert demanda[8] == 2
    assert demanda[18] == 1
    assert demanda[3] == 0


def test_demanda_por_dia_semana():
    eda = ViajesEDA(fixture_viajes_prueba())
    resumen = eda.obtener_demanda_por_dia_semana()
    assert len(resumen) == 7
    lunes = resumen.loc[resumen["dia_semana"] == "Lunes"]
    assert lunes["total_viajes"].values[0] == 2


def test_matriz_dia_hora():
    eda = ViajesEDA(fixture_viajes_prueba())
    matriz = eda.obtener_matriz_dia_hora()
    assert matriz.shape == (7, 24)
    assert matriz.loc["Lunes", 8] == 2.0


def test_top_estaciones_y_balance():
    eda = ViajesEDA(fixture_viajes_prueba(), fixture_nomenclatura_prueba())
    salidas, llegadas = eda.obtener_top_estaciones(top_n=2)
    assert salidas.iloc[0]["id"] == 10
    assert salidas.iloc[0]["total_salidas"] == 2
    assert llegadas.iloc[0]["id"] in [10, 20]

    balance = eda.obtener_balance_estaciones()
    # Estacion 20: 1 salida, 2 llegadas -> balance neto +1
    est_20 = balance.loc[balance["id"] == 20].iloc[0]
    assert est_20["balance_neto"] == 1


def test_metricas_adicionales_eda():
    eda = ViajesEDA(fixture_viajes_prueba(), fixture_nomenclatura_prueba())
    
    # Probar duración visibles
    visibles, limite, fuera = eda.obtener_duraciones_visibles(quantil=1.0)
    assert len(visibles) == 4
    assert fuera == 0
    
    # Probar distribución género
    dist_gen = eda.obtener_distribucion_genero()
    assert len(dist_gen) == 2
    assert dist_gen.loc[dist_gen["genero"] == "M", "total_viajes"].iloc[0] == 2
    
    # Probar MLflow dict
    metricas = eda.obtener_metricas_mlflow()
    assert metricas["total_viajes"] == 4
    assert metricas["viajes_sin_edad"] == 1
