from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# data/ no se sube a Git, así que al clonar el repo estas carpetas no existen: las creamos aquí
for carpeta in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR):
    carpeta.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow: registro propio del proyecto. La base SQLite y los artefactos viven en la raíz de
# mibici-demand (ambos en .gitignore). Usamos rutas absolutas para que funcione igual desde
# la raíz, desde notebooks/ o desde tests/.
MLFLOW_TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "mibici-fase1"


def configurar_mlflow():
    """Apunta MLflow al registro del proyecto y activa el experimento de la fase 1."""
    import mlflow  # se importa aquí para no hacer lento cualquier `import mibici.config`

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
