
"""Pruebas de persistencia de checkpoints del modelado MiBici."""

import pandas as pd
import json

import pytest

from mibici.modeling.checkpoints import (
    cargar_checkpoint,
    guardar_checkpoint,
    identificar_ejecucion,
)


def test_whenCheckpointIsSaved_thenCanBeRecovered(tmp_path):
    """Recupera el contenido previamente guardado."""
    ruta = tmp_path / "checkpoint.json"

    contenido = {
        "ventana": 1,
        "mes_prueba": "2026-04",
        "mae": 0.85,
        "parametros": {
            "loss": "poisson",
            "max_iter": 200,
        },
    }

    guardar_checkpoint(ruta, contenido)

    assert cargar_checkpoint(ruta) == contenido


def test_whenCheckpointDoesNotExist_thenReturnsNone(tmp_path):
    """Devuelve None cuando el checkpoint no existe."""
    ruta = tmp_path / "inexistente.json"

    assert cargar_checkpoint(ruta) is None


def test_whenCheckpointAlreadyExists_thenReplacesContent(tmp_path):
    """Actualiza un checkpoint existente."""
    ruta = tmp_path / "checkpoint.json"

    guardar_checkpoint(ruta, {"ventana": 1})
    guardar_checkpoint(ruta, {"ventana": 2})

    assert cargar_checkpoint(ruta) == {"ventana": 2}


def test_whenParentDirectoryDoesNotExist_thenCreatesIt(tmp_path):
    """Crea los directorios necesarios."""
    ruta = tmp_path / "modelado" / "hgb" / "checkpoint.json"

    guardar_checkpoint(ruta, {"estado": "completado"})

    assert ruta.exists()
    assert cargar_checkpoint(ruta) == {"estado": "completado"}


def test_whenCheckpointIsSaved_thenDoesNotLeaveTemporaryFiles(tmp_path):
    """El guardado exitoso no deja archivos temporales."""
    ruta = tmp_path / "checkpoint.json"

    guardar_checkpoint(ruta, {"mae": 0.8})

    assert sorted(archivo.name for archivo in tmp_path.iterdir()) == [
        "checkpoint.json"
    ]


def test_whenCheckpointContainsInvalidStructure_thenRaisesValueError(
    tmp_path,
):
    """Rechaza checkpoints cuyo contenido no sea un objeto JSON."""
    ruta = tmp_path / "checkpoint.json"
    ruta.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="objeto JSON"):
        cargar_checkpoint(ruta)


def test_whenCheckpointContainsInvalidJson_thenRaisesDecodeError(
    tmp_path,
):
    """Rechaza archivos JSON corruptos."""
    ruta = tmp_path / "checkpoint.json"
    ruta.write_text("{contenido_invalido", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        cargar_checkpoint(ruta)


def test_whenSavingInvalidValue_thenPreservesPreviousCheckpoint(
    tmp_path,
):
    """Un error de serialización no destruye el checkpoint anterior."""
    ruta = tmp_path / "checkpoint.json"
    guardar_checkpoint(ruta, {"estado": "válido"})

    with pytest.raises(ValueError):
        guardar_checkpoint(ruta, {"mae": float("nan")})

    assert cargar_checkpoint(ruta) == {"estado": "válido"}

    assert sorted(archivo.name for archivo in tmp_path.iterdir()) == [
        "checkpoint.json"
    ]



@pytest.fixture
def datos_identificacion():
    """Datos y configuración de una ejecución de entrenamiento."""
    datos = pd.DataFrame(
        {
            "estacion_id": [101, 101, 202],
            "salidas_lag_168": [5.0, 8.0, 12.0],
        }
    )

    objetivo = [10, 15, 20]

    configuracion = {
        "modelo": "hist_gradient_boosting",
        "semilla": 42,
        "n_iter": 8,
    }

    return datos, objetivo, configuracion


def test_whenExecutionIsIdentical_thenGeneratesSameIdentifier(
    datos_identificacion,
):
    datos, objetivo, configuracion = datos_identificacion

    primero = identificar_ejecucion(datos, objetivo, configuracion)
    segundo = identificar_ejecucion(datos, objetivo, configuracion)

    assert primero == segundo
    assert len(primero) == 64


def test_whenDatasetChanges_thenGeneratesDifferentIdentifier(
    datos_identificacion,
):
    datos, objetivo, configuracion = datos_identificacion

    original = identificar_ejecucion(datos, objetivo, configuracion)

    modificados = datos.copy()
    modificados.loc[0, "salidas_lag_168"] = 99.0

    nuevo = identificar_ejecucion(
        modificados, objetivo, configuracion
    )

    assert original != nuevo


def test_whenTargetChanges_thenGeneratesDifferentIdentifier(
    datos_identificacion,
):
    datos, objetivo, configuracion = datos_identificacion

    original = identificar_ejecucion(datos, objetivo, configuracion)

    nuevo = identificar_ejecucion(
        datos, [10, 15, 99], configuracion
    )

    assert original != nuevo


def test_whenConfigurationChanges_thenGeneratesDifferentIdentifier(
    datos_identificacion,
):
    datos, objetivo, configuracion = datos_identificacion

    original = identificar_ejecucion(datos, objetivo, configuracion)

    nueva_configuracion = {
        **configuracion,
        "n_iter": 2,
    }

    nuevo = identificar_ejecucion(
        datos, objetivo, nueva_configuracion
    )

    assert original != nuevo


def test_whenConfigurationKeyOrderChanges_thenIdentifierIsSame(
    datos_identificacion,
):
    datos, objetivo, configuracion = datos_identificacion

    original = identificar_ejecucion(datos, objetivo, configuracion)

    invertida = dict(reversed(list(configuracion.items())))

    nuevo = identificar_ejecucion(datos, objetivo, invertida)

    assert original == nuevo
