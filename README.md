# mibici-demand

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pronóstico de demanda de bicicletas por estación y hora (MiBici GDL)

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Descarga de datos

Los datos vienen de los [datos abiertos de MiBici](https://mibici.net/en/open-data/) y **no se
suben a Git**. Para bajarlos a `data/raw/` (con el `.venv` activado):

```bash
# Rango por defecto: 2025-10 a 2026-09
python mibici/dataset.py

# Otro rango de meses (formato AAAA-MM, ambos inclusive)
python mibici/dataset.py --inicio 2025-01 --fin 2025-06

# Equivalente con make (usa el rango por defecto)
make data
```

- Baja un CSV de viajes por mes (`datos_abiertos_AAAA_MM.csv`) y el CSV de nomenclatura de
  estaciones (`nomenclatura_AAAA_MM.csv`).
- Los enlaces se extraen de la página de datos abiertos, porque no siguen un patrón fijo.
- Omite los archivos que ya están descargados, así que se puede volver a correr sin problema.
- Avisa en consola qué meses del rango no tienen enlace (por ejemplo, meses aún no publicados).
- Las carpetas de `data/` se crean solas la primera vez (`mibici/config.py`).
- También se puede usar desde Python: `from mibici.dataset import descargar_meses`.

## Limpieza y versionado de datos con MLflow

```bash
python -m mibici.cleaning                                  # limpia y registra en MLflow
mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db     # abrir http://127.0.0.1:5000
```

El script lee todos los CSV de `data/raw/`, aplica la clase `DataCleaner` (`mibici/cleaning.py`)
y guarda el resultado en `data/interim/viajes_limpios.parquet`. Todo queda en el experimento
**`mibici-fase1`** de MLflow (base `mlflow.db` y artefactos en `mlruns/`, ambos fuera de Git):

- Dataset crudo **`mibici_raw_v1`** (`context="raw"`) y dataset limpio **`mibici_interim_v1`**
  (`context="interim"`), con su esquema, número de filas y *digest* para detectar cambios.
- Parámetros: meses procesados y filas antes y después.
- Artefacto `bitacora_limpieza.csv`: por cada regla, filas antes, después y afectadas.
- Artefacto `datos/viajes_limpios.parquet`: copia del dataset limpio (~96 MB por corrida). El
  crudo no se guarda porque se reconstruye con `mibici/dataset.py`.
- Tags que ligan cada versión de los datos con su origen: `git_commit` (hash del código que la
  generó), `git_cambios_sin_commit` (`true` si se corrió con cambios sin commit, en cuyo caso
  el hash no describe exactamente el código) y `fuente_datos` (URL de los datos abiertos).

Para que `git_commit` sea confiable, haz commit de tus cambios **antes** de correr la limpieza.

### Reglas, en el orden en que se aplican

Los números vienen del diagnóstico (`reports/diagnostico_raw.md`). En la lectura, cada archivo
se abre como UTF-8 y, si falla, como Latin-1 (caso de 2026-01).

| # | Regla (método) | Qué hace | Por qué |
|---|---|---|---|
| 1 | `eliminar_duplicados` | Quita filas idénticas | 758 filas repetidas en 2026-09; contarían doble la demanda |
| 2 | `parsear_fechas` | Inicio y fin a datetime, agrega `duracion_min` | Las reglas de duración y el conteo por hora lo necesitan |
| 3 | `eliminar_duracion_cero` | Quita viajes de 0 segundos | Terminan en el mismo instante en que empiezan: no son viajes |
| 4 | `eliminar_cortos_misma_estacion` | Quita viajes < 1 min con origen = destino | La bici se sacó y se devolvió de inmediato (anclaje fallido o arrepentimiento); no es demanda real |
| 5 | `eliminar_mas_de_24h` | Quita viajes de más de 24 h | Bicicletas no devueltas o errores de registro (13 viajes, máximo 98 h) |
| 6 | `marcar_duracion_atipica` | `duracion_atipica = True` si dura de 3 a 24 h | Son raros (p99.9 = 59 min) pero la salida sí ocurrió; se conservan y se pueden excluir en el EDA |
| 7 | `anio_nacimiento_invalido_a_nulo` | Año < 1920 pasa a nulo | Años imposibles (199, 1420); el viaje es válido, solo el dato del usuario no |
| 8 | `genero_nulo_a_desconocido` | Género vacío pasa a `"desconocido"` | Conserva los viajes (0.7%) y deja explícita la categoría |
| 9 | `marcar_estaciones_en_catalogo` | `estacion_en_catalogo = True` si origen y destino están en la nomenclatura | Las estaciones 405 y 407 no están en el catálogo pero tuvieron viajes reales (junio y julio 2026); se conservan |

En las reglas que marcan o corrigen, "filas afectadas" en la bitácora cuenta las filas
modificadas; en la regla 9 cuenta los viajes **fuera** del catálogo.

Resultado con 2025-10 a 2026-09: **4,497,877 → 4,354,416 viajes** (se elimina el 3.2%).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

