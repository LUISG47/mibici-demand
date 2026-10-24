"""
Pruebas unitarias para mibici.diagnostics.
"""

from pathlib import Path
import pandas as pd

from mibici.diagnostics import RawDataDiagnoser


def fixture_datos_de_viajes() -> pd.DataFrame:
    """Genera un DataFrame sintético con varios escenarios de viajes."""
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3, 4, 5, 5],  # Incluye un duplicado intencional
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                    "2026-01-01 13:00:00",
                    "2026-01-01 14:00:00",
                    "2026-01-01 14:00:00",
                ]
            ),
            "Fin_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 10:00:30",  # < 1 min, misma estación
                    "2026-01-01 11:15:00",  # 15 min, normal
                    "2026-01-01 16:30:00",  # 4.5 h (> 3 h)
                    "2026-01-01 13:00:00",  # 0 seg, misma estación
                    "2026-01-02 15:00:00",  # > 24 h
                    "2026-01-02 15:00:00",
                ]
            ),
            "Origen_Id": [10, 10, 20, 30, 999, 999],
            "Destino_Id": [10, 20, 20, 30, 20, 20],
            "Genero": ["M", "F", None, "M", "F", "F"],
            "Año_de_nacimiento": [1990.0, 1985.0, 1900.0, None, 2000.0, 2000.0],
        }
    )


def fixture_nomenclatura_estaciones() -> pd.DataFrame:
    """Genera una tabla sintética de estaciones con casos borde."""
    return pd.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "name": [
                "Estación A",
                "Estación B",
                "Estación C",
                "ex-Estación Inactiva",
            ],
            "dpcapacity": [15, 20, 10, 0],
        }
    )


def test_extraer_mes_de_ruta():
    ruta = Path("data/raw/datos_abiertos_2026_09.csv")
    mes = RawDataDiagnoser.extraer_mes_de_ruta(ruta)
    assert mes == "2026-09"


def test_dataframe_a_markdown():
    df = pd.DataFrame(
        {
            "col_str": ["a", "b"],
            "col_int": [10, 20],
            "col_float": [1.234, 5.678],
            "col_bool": [True, False],
        }
    )
    md = RawDataDiagnoser.dataframe_a_markdown(df)

    assert "| col_str | col_int | col_float | col_bool |" in md
    assert "| a | 10 | 1.23 | sí |" in md
    assert "| b | 20 | 5.68 | no |" in md


def test_computar_metricas_de_duracion():
    datos = fixture_datos_de_viajes()
    diagnosticador = RawDataDiagnoser(datos)
    resultado = diagnosticador.computar_metricas_de_duracion()

    conteos = resultado["counts"]
    assert conteos.loc["cero", "viajes"] == 1
    # Contabiliza el viaje de 30s y el de 0s en misma estación
    assert conteos.loc["menor a 1 min y misma estación", "viajes"] == 2
    assert conteos.loc["mayor a 3 h", "viajes"] == 3
    assert conteos.loc["mayor a 24 h", "viajes"] == 2
    assert conteos.loc["negativa", "viajes"] == 0

    percentiles = resultado["percentiles"]
    assert "minutos" in percentiles.columns
    assert percentiles.loc["máximo", "minutos"] > 1440.0


def test_viaje_duracion_cero_misma_estacion():
    df_cero = pd.DataFrame(
        {
            "Viaje_Id": [100],
            "Inicio_del_viaje": pd.to_datetime(["2026-01-01 08:00:00"]),
            "Fin_del_viaje": pd.to_datetime(["2026-01-01 08:00:00"]),
            "Origen_Id": [15],
            "Destino_Id": [15],
        }
    )
    diagnosticador = RawDataDiagnoser(df_cero)
    resultado = diagnosticador.computar_metricas_de_duracion()
    conteos = resultado["counts"]

    assert conteos.loc["cero", "viajes"] == 1
    assert conteos.loc["menor a 1 min y misma estación", "viajes"] == 1


def test_revisar_consistencia_de_estaciones():
    viajes = fixture_datos_de_viajes()
    nomenclatura = fixture_nomenclatura_estaciones()
    diagnosticador = RawDataDiagnoser(viajes, nomenclatura)
    resultado = diagnosticador.revisar_consistencia_de_estaciones()

    faltantes = resultado["missing_stations"]
    assert 999 in faltantes.index
    assert faltantes.loc[999, "viajes_como_origen"] == 2
    assert faltantes.loc[999, "viajes_como_destino"] == 0

    no_utilizadas = resultado["unused_stations"]
    assert 40 in no_utilizadas["id"].values


def test_flujo_completo_ejecutar():
    viajes = fixture_datos_de_viajes()
    nomenclatura = fixture_nomenclatura_estaciones()
    diagnosticador = RawDataDiagnoser(viajes, nomenclatura)
    resultados = diagnosticador.ejecutar()

    # Valida duplicados
    assert resultados["duplicados"] == 1

    # Valida nulos
    nulos = resultados["nulos"]
    assert "columna" in nulos.columns
    assert nulos.loc[nulos["columna"] == "Genero", "nulos"].values[0] == 1

    # Valida demográficos
    demograficos = resultados["demograficos"]
    assert demograficos["anios_invalidos"] == 1  # año 1900 < 1920

    # Valida rango de fechas
    fechas = resultados["rango_fechas"]
    assert fechas["total_dias"] >= 1
