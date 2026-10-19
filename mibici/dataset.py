"""Descarga de los datos abiertos de MiBici: viajes mensuales y nomenclatura de estaciones."""

from pathlib import Path
import re
from urllib.parse import urljoin

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici.config import RAW_DATA_DIR

URL_DATOS_ABIERTOS = "https://mibici.net/en/open-data/"

# El servidor (ModSecurity) rechaza User-Agents genéricos como el de requests,
# así que nos identificamos como un navegador común
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
}

app = typer.Typer()


def obtener_enlaces() -> tuple[dict[str, str], str | None]:
    """Extrae de la página de datos abiertos los enlaces a los CSV.

    Regresa un diccionario {"AAAA-MM": url} con los viajes mensuales
    y la url del CSV de nomenclatura de estaciones (o None si no aparece).
    """
    respuesta = requests.get(URL_DATOS_ABIERTOS, headers=HEADERS, timeout=30)
    respuesta.raise_for_status()

    meses = {}
    nomenclatura = None
    # Los nombres no siguen un patrón fijo (p. ej. datos_abiertos_2016_01-1.csv),
    # por eso buscamos todos los .csv y sacamos el año y mes del nombre
    for href in re.findall(r'href="([^"]+\.csv)"', respuesta.text):
        url = urljoin(URL_DATOS_ABIERTOS, href)
        fecha = re.search(r"datos_abiertos_(\d{4})_(\d{2})", href)
        if fecha:
            meses.setdefault(f"{fecha[1]}-{fecha[2]}", url)
        elif "nomenclatura" in href:
            nomenclatura = url
    return meses, nomenclatura


def descargar_archivo(url: str, destino: Path) -> Path:
    """Descarga un archivo a `destino`, salvo que ya exista."""
    if destino.exists():
        logger.info(f"Ya existe, se omite: {destino.name}")
        return destino

    # Escribimos primero a un archivo .part para no dejar descargas a medias
    # con el nombre final si algo falla
    temporal = destino.with_suffix(".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as respuesta:
        respuesta.raise_for_status()
        total = int(respuesta.headers.get("content-length", 0))
        with (
            open(temporal, "wb") as archivo,
            tqdm(total=total, unit="B", unit_scale=True, desc=destino.name) as barra,
        ):
            for bloque in respuesta.iter_content(chunk_size=1024 * 1024):
                archivo.write(bloque)
                barra.update(len(bloque))
    temporal.rename(destino)
    return destino


def descargar_meses(inicio: str, fin: str, carpeta: Path = RAW_DATA_DIR) -> list[Path]:
    """Descarga los viajes de los meses entre `inicio` y `fin` (formato AAAA-MM, inclusive)
    y el CSV de nomenclatura de estaciones. Regresa las rutas de los archivos."""
    meses_pedidos = pd.period_range(inicio, fin, freq="M").strftime("%Y-%m")
    enlaces, url_nomenclatura = obtener_enlaces()

    archivos = []
    sin_enlace = []
    for mes in meses_pedidos:
        if mes not in enlaces:
            sin_enlace.append(mes)
            continue
        # Guardamos con un nombre uniforme, sin sufijos como "-1"
        nombre = f"datos_abiertos_{mes.replace('-', '_')}.csv"
        archivos.append(descargar_archivo(enlaces[mes], carpeta / nombre))

    if sin_enlace:
        logger.warning(f"Meses sin enlace en la página: {', '.join(sin_enlace)}")

    if url_nomenclatura:
        nombre = url_nomenclatura.rsplit("/", 1)[-1]
        archivos.append(descargar_archivo(url_nomenclatura, carpeta / nombre))
    else:
        logger.warning("No se encontró el enlace de nomenclatura de estaciones")

    return archivos


@app.command()
def main(
    inicio: str = typer.Option("2025-10", help="Primer mes a descargar (AAAA-MM)"),
    fin: str = typer.Option("2026-09", help="Último mes a descargar (AAAA-MM)"),
):
    """Descarga los datos abiertos de MiBici a data/raw/."""
    archivos = descargar_meses(inicio, fin)
    logger.success(f"{len(archivos)} archivos listos en {RAW_DATA_DIR}")


if __name__ == "__main__":
    app()
