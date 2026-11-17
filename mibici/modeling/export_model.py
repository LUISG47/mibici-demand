"""Exportación del modelo registrado en MLflow a models/ para servirlo (API y Docker)."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
import typer

from mibici.cleaning import estado_git
from mibici.config import MLFLOW_TRACKING_URI, MODELS_DIR, cargar_parametros

app = typer.Typer()


def sha256_archivo(ruta: Path) -> str:
    """Huella SHA-256 de un archivo, para verificar que el binario no cambió."""
    huella = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


def construir_metadata(nombre: str, version: str, destino: Path) -> dict:
    """Reúne la trazabilidad del modelo: Registry, run de entrenamiento y exportación."""
    cliente = mlflow.MlflowClient()
    version_modelo = cliente.get_model_version(nombre, version)
    run = cliente.get_run(version_modelo.run_id)

    # Las 20 columnas de entrada salen de la firma guardada con el modelo
    firma = mlflow.models.get_model_info(str(destino)).signature
    commit_exportacion, cambios_sin_commit = estado_git()

    return {
        "modelo_uri": f"models:/{nombre}/{version}",
        "modelo_registrado": nombre,
        "version": int(version),
        "aliases": sorted(version_modelo.aliases),
        "run_id": version_modelo.run_id,
        "metricas_promedio": run.data.metrics,
        "hiperparametros": run.data.params,
        "git_commit": run.data.tags.get("git_commit", "desconocido"),
        "git_commit_exportacion": commit_exportacion,
        "git_cambios_sin_commit_exportacion": cambios_sin_commit,
        "fecha_exportacion": datetime.now(UTC).isoformat(timespec="seconds"),
        "columnas_entrada": firma.inputs.input_names(),
        "sha256_model_pkl": sha256_archivo(destino / "model.pkl"),
        "mlflow_version": mlflow.__version__,
    }


def exportar_modelo(
    nombre: str | None = None,
    version: int | None = None,
    destino: Path | None = None,
    sobrescribir: bool = False,
) -> Path:
    """Copia models:/<nombre>/<version> como modelo MLflow a models/ y agrega metadata.json.

    Por defecto el nombre, la versión y la carpeta se leen de la sección serving: de
    params.yaml. Regresa la carpeta del modelo exportado.
    """
    serving = cargar_parametros()["serving"]
    nombre = nombre or serving["modelo_registrado"]
    version = str(version or serving["version"])
    destino = Path(destino or MODELS_DIR / serving["directorio_modelo"])

    if destino.exists():
        if not sobrescribir:
            raise FileExistsError(f"Ya existe {destino}; usa --sobrescribir para reemplazarlo.")
        shutil.rmtree(destino)

    # Solo se apunta al registro del proyecto: exportar no debe crear experimentos
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    uri = f"models:/{nombre}/{version}"
    logger.info(f"Exportando {uri} a {destino}")
    destino.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(destino))

    metadata = construir_metadata(nombre, version, destino)
    with open(destino / "metadata.json", "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")

    logger.success(f"Modelo exportado en {destino} (run_id {metadata['run_id']})")
    return destino


@app.command()
def main(
    nombre: str | None = None,
    version: int | None = None,
    destino: Path | None = None,
    sobrescribir: bool = False,
):
    """Exporta el modelo registrado indicado en params.yaml (sección serving:) a models/."""
    exportar_modelo(nombre=nombre, version=version, destino=destino, sobrescribir=sobrescribir)


if __name__ == "__main__":
    app()
