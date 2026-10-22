"""
Pruebas unitarias para mibici.diagnostics.
"""

from pathlib import Path
import pandas as pd
import pytest

from mibici.diagnostics import RawDataDiagnoser


@pytest.fixture
def sample_trips_data() -> pd.DataFrame:
    """
    Genera un DataFrame sintético con varios escenarios de viajes.
    """
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3, 4, 5],
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                    "2026-01-01 13:00:00",
                    "2026-01-01 14:00:00",
                ]
            ),
            "Fin_del_viaje": pd.to_datetime(
                [
                    "2026-01-01 10:00:30",  # < 1 min, misma estación
                    "2026-01-01 11:15:00",  # 15 min, normal
                    "2026-01-01 16:30:00",  # 4.5 h (> 3 h)
                    "2026-01-01 13:00:00",  # 0 seg
                    "2026-01-02 15:00:00",  # > 24 h
                ]
            ),
            "Origen_Id": [10, 10, 20, 30, 999],
            "Destino_Id": [10, 20, 20, 30, 20],
        }
    )


@pytest.fixture
def sample_nomenclature() -> pd.DataFrame:
    """
    Genera una tabla sintética de estaciones con casos borde.
    """
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


def test_extract_month_from_path():
    path = Path("data/raw/datos_abiertos_2026_09.csv")
    month = RawDataDiagnoser.extract_month_from_path(path)
    assert month == "2026-09"


def test_dataframe_to_markdown():
    df = pd.DataFrame(
        {
            "col_str": ["a", "b"],
            "col_int": [10, 20],
            "col_float": [1.234, 5.678],
            "col_bool": [True, False],
        }
    )
    md = RawDataDiagnoser.dataframe_to_markdown(df)

    assert "| col_str | col_int | col_float | col_bool |" in md
    assert "| a | 10 | 1.23 | sí |" in md
    assert "| b | 20 | 5.68 | no |" in md


def test_compute_duration_metrics(sample_trips_data):
    diagnoser = RawDataDiagnoser()
    result = diagnoser.compute_duration_metrics(sample_trips_data)

    assert "counts" in result
    assert "percentiles" in result

    counts = result["counts"]
    assert counts.loc["cero", "viajes"] == 1
    assert counts.loc["menor a 1 min y misma estación", "viajes"] == 1
    assert counts.loc["mayor a 3 h", "viajes"] == 2
    assert counts.loc["mayor a 24 h", "viajes"] == 1
    assert counts.loc["negativa", "viajes"] == 0

    percentiles = result["percentiles"]
    assert "minutos" in percentiles.columns
    assert percentiles.loc["máximo", "minutos"] > 1440.0


def test_check_station_consistency(sample_trips_data, sample_nomenclature):
    diagnoser = RawDataDiagnoser()
    result = diagnoser.check_station_consistency(sample_trips_data, sample_nomenclature)

    missing = result["missing_stations"]
    assert 999 in missing.index
    assert missing.loc[999, "viajes_como_origen"] == 1
    assert missing.loc[999, "viajes_como_destino"] == 0

    unused = result["unused_stations"]
    assert 40 in unused["id"].values
