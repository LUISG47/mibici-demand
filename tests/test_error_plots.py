
"""Pruebas de visualización de errores del modelo MiBici."""

from pathlib import Path

import pandas as pd
import pytest

from mibici.modeling.error_plots import generar_figuras_diagnostico


@pytest.fixture
def diagnostico_prueba():
    """Crea predicciones de ejemplo para distintas estaciones y horas."""
    return pd.DataFrame(
        {
            "estacion_id": [3, 71, 158, 266, 3, 71],
            "hora": [1, 1, 1, 1, 18, 18],
            "valor_real": [0, 0, 1, 0, 3, 2],
            "prediccion": [100, 200, 300, 400, 2, 3],
            "error_cuadratico": [
                10000, 40000, 89401, 160000, 1, 1
            ],
        }
    )


def test_whenValidDiagnosticsAreProvided_thenGeneratesThreeFigures(
    diagnostico_prueba,
    tmp_path,
):
    rutas = generar_figuras_diagnostico(
        diagnostico_prueba,
        tmp_path / "figures",
    )

    assert len(rutas) == 3

    nombres_esperados = {
        "hgb_rmse_por_hora_junio.png",
        "hgb_error_por_estacion_junio.png",
        "hgb_predicciones_extremas_junio.png",
    }

    assert {ruta.name for ruta in rutas} == nombres_esperados

    for ruta in rutas:
        assert isinstance(ruta, Path)
        assert ruta.is_file()
        assert ruta.stat().st_size > 0


def test_whenRequiredColumnIsMissing_thenRaisesValueError(
    diagnostico_prueba,
    tmp_path,
):
    datos = diagnostico_prueba.drop(columns=["error_cuadratico"])

    with pytest.raises(
        ValueError,
        match="Faltan columnas requeridas",
    ):
        generar_figuras_diagnostico(
            datos,
            tmp_path / "figures",
        )


def test_whenOutputDirectoryDoesNotExist_thenCreatesIt(
    diagnostico_prueba,
    tmp_path,
):
    directorio = tmp_path / "nuevo" / "figures"

    assert not directorio.exists()

    rutas = generar_figuras_diagnostico(
        diagnostico_prueba,
        directorio,
    )

    assert directorio.is_dir()
    assert len(rutas) == 3
