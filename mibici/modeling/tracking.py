"""Registro del modelado en MLflow: runs por modelo, runs anidados por ventana y Registry."""

from pathlib import Path
import tempfile

import mlflow
from mlflow.models import infer_signature

from mibici.cleaning import estado_git
from mibici.config import MLFLOW_ARTIFACTS_DIR, MLFLOW_TRACKING_URI, cargar_parametros
from mibici.dataset import URL_DATOS_ABIERTOS


def configurar_mlflow(experimento=None):
    """Apunta MLflow al registro del proyecto y activa el experimento de modelado."""
    experimento = experimento or cargar_parametros()["mlflow"]["experimento"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri())

    mlflow.set_experiment(experimento)


def registrar_evaluacion(
    nombre_modelo,
    tabla,
    resumen,
    hiperparametros,
    modelo_final=None,
    X_ejemplo=None,
    fuente_datos=URL_DATOS_ABIERTOS,
):
    """Registra la evaluación walk-forward de un modelo y regresa el run_id del padre.

    - Run padre: tags de trazabilidad, hiperparámetros, métricas promedio y la tabla por
      ventana como artefacto. Si se pasa modelo_final, se guarda con firma y ejemplo.
    - Un run anidado por ventana con sus métricas.
    """
    commit, cambios_sin_commit = estado_git()

    with mlflow.start_run(run_name=nombre_modelo) as padre:
        mlflow.set_tags(
            {
                "git_commit": commit,
                "git_cambios_sin_commit": str(cambios_sin_commit).lower(),
                "fuente_datos": fuente_datos,
            }
        )
        mlflow.log_params(hiperparametros)
        mlflow.log_metrics(resumen)

        # Tabla por ventana como artefacto CSV
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "tabla_ventanas.csv"
            tabla.to_csv(ruta, index=False)
            mlflow.log_artifact(str(ruta), artifact_path="evaluacion")

        # Un run anidado por ventana
        for fila in tabla.to_dict("records"):
            with mlflow.start_run(run_name=f"ventana_{fila['mes_prueba']}", nested=True):
                mlflow.log_params(
                    {
                        "mes_prueba": fila["mes_prueba"],
                        "n_entrenamiento": fila["n_entrenamiento"],
                        "n_prueba": fila["n_prueba"],
                    }
                )
                mlflow.log_metrics({m: fila[m] for m in ["mae", "rmse", "wape"]})

        # Modelo final (lo entrena train.py); solo se guarda si se pasa
        if modelo_final is not None:
            if X_ejemplo is None:
                raise ValueError("Para guardar el modelo se necesita X_ejemplo (firma).")

            ejemplo = X_ejemplo.head(5)
            firma = infer_signature(ejemplo, modelo_final.predict(ejemplo))

            mlflow.sklearn.log_model(
                modelo_final, name="modelo", signature=firma, input_example=ejemplo
            )

    return padre.info.run_id


def registrar_modelo(
    run_id, hiperparametros, metricas, nombre=None, alias=None, ruta_artefacto="modelo"
):
    """Registra el modelo del run en el Model Registry y regresa la versión creada.

    Los hiperparámetros y las métricas promedio quedan como tags de la versión. Si se pasa
    alias (por ejemplo "champion"), se asigna a la versión para cargarla después como
    models:/<nombre>@<alias>.
    """
    nombre = nombre or cargar_parametros()["mlflow"]["modelo_registrado"]

    version = mlflow.register_model(f"runs:/{run_id}/{ruta_artefacto}", nombre)

    cliente = mlflow.MlflowClient()

    for clave, valor in {**hiperparametros, **metricas}.items():
        cliente.set_model_version_tag(nombre, version.version, clave, str(valor))

    if alias:
        cliente.set_registered_model_alias(nombre, alias, version.version)

    return version
