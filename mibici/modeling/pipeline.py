"""Pipeline de preprocesamiento y modelado de MiBici."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from mibici.config import cargar_parametros


def crear_pipeline(estimador, params=None):
    """Construye un Pipeline de Scikit-Learn para predecir salidas."""

    if params is None:
        params = cargar_parametros()

    rezagos = params["features"]["rezagos"]
    semilla = params["modelado"]["semilla"]
    particiones_cv = params["modelado"]["target_encoder_cv"]

    # Variables históricas sin normalizar del dataset v3
    columnas_historicas = [
        f"{tipo}_lag_{lag}" for tipo in ["salidas", "llegadas"] for lag in rezagos
    ]

    # Variables temporales ya representadas de forma cíclica
    columnas_temporales = [
        "hora_sin",
        "hora_cos",
        "dia_semana_sin",
        "dia_semana_cos",
        "mes_sin",
        "mes_cos",
        "fin_semana",
        "festivo",
    ]

    # Imputación y escalado ajustados con entrenamiento
    transformar_historicas = Pipeline(
        steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]
    )

    # Combinar las transformaciones
    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "historicas",
                transformar_historicas,
                columnas_historicas,
            ),
            (
                "estacion",
                TargetEncoder(
                    target_type="continuous",
                    cv=KFold(
                        n_splits=particiones_cv,
                        shuffle=True,
                        random_state=semilla,
                    ),
                ),
                ["estacion_id"],
            ),
            (
                "temporales",
                "passthrough",
                columnas_temporales,
            ),
        ],
        remainder="drop",
    )

    # El estimador se recibe desde train.py
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", estimador),
        ]
    )
