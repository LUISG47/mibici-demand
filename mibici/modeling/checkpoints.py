
"""Persistencia atómica de checkpoints para entrenamientos de MiBici."""

import hashlib

import pandas as pd
import json
import os
from pathlib import Path
import tempfile


def guardar_checkpoint(ruta: Path, contenido: dict) -> None:
    """Guarda un checkpoint sin sobrescribirlo parcialmente."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    temporal = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=ruta.parent,
            prefix=f".{ruta.name}.",
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            json.dump(contenido, archivo, indent=2, allow_nan=False)
            archivo.flush()
            os.fsync(archivo.fileno())

        os.replace(temporal, ruta)

    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def cargar_checkpoint(ruta: Path) -> dict | None:
    """Recupera un checkpoint si existe y contiene un objeto JSON."""
    ruta = Path(ruta)

    if not ruta.exists():
        return None

    with ruta.open(encoding="utf-8") as archivo:
        contenido = json.load(archivo)

    if not isinstance(contenido, dict):
        raise ValueError(
            f"El checkpoint debe contener un objeto JSON: {ruta}"
        )

    return contenido


def identificar_ejecucion(
    datos: pd.DataFrame,
    objetivo,
    configuracion: dict,
) -> str:
    """Identifica una ejecución según sus datos y configuración."""

    digest = hashlib.sha256()

    digest.update(
        json.dumps(
            configuracion,
            sort_keys=True,
            default=str,
            allow_nan=False,
        ).encode("utf-8")
    )

    digest.update(
        repr(
            [
                (columna, str(tipo))
                for columna, tipo in datos.dtypes.items()
            ]
        ).encode("utf-8")
    )

    digest.update(str(datos.shape).encode("utf-8"))

    # Procesar en bloques para limitar el uso adicional de memoria.
    for inicio in range(0, len(datos), 100_000):
        bloque = datos.iloc[inicio : inicio + 100_000]

        hashes = pd.util.hash_pandas_object(
            bloque,
            index=False,
        ).to_numpy()

        digest.update(hashes.tobytes())

    serie_objetivo = pd.Series(objetivo)

    digest.update(str(len(serie_objetivo)).encode("utf-8"))

    for inicio in range(0, len(serie_objetivo), 100_000):
        bloque = serie_objetivo.iloc[inicio : inicio + 100_000]

        hashes = pd.util.hash_pandas_object(
            bloque,
            index=False,
        ).to_numpy()

        digest.update(hashes.tobytes())

    return digest.hexdigest()

