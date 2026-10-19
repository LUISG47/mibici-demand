# Cómo contribuir

Flujo de trabajo del Equipo 1 para `mibici-demand`. Las reglas completas del proyecto están en
`CLAUDE.md`.

## 1. Preparar el entorno

```bash
git clone https://github.com/LUISG47/mibici-demand.git
cd mibici-demand
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` instala también el paquete `mibici` en modo editable (`-e .`).

## 2. Obtener los datos

Los datos no están en Git. Cada integrante los genera en su máquina:

```bash
python mibici/dataset.py        # descarga 2025-10 a 2026-09 a data/raw/
python -m mibici.cleaning       # limpia a data/interim/ y registra la versión en MLflow
mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db   # opcional: ver los runs
```

Cada quien tiene su propio `mlflow.db` local. Los digests de los datasets permiten comprobar
que todos trabajamos con los mismos datos.

## 3. Trabajar en una rama

```bash
git switch main
git pull
git switch -c feature/<tema>     # p. ej. feature/eda, feature/preprocesamiento
```

- Una rama por tema, ligada a su Issue cuando exista.
- Commits descriptivos **en español**: un título corto en imperativo y, si hace falta, el
  detalle en el cuerpo. Ejemplo: `Agrega conteos de salidas por estación y hora`.
- Antes de subir cambios: `make lint` y `make test` sin errores.
- Para que el tag `git_commit` de MLflow sea confiable, haz commit **antes** de correr un
  script que registre datos.

## 4. Pull Request

```bash
git push -u origin feature/<tema>
```

Abre el Pull Request hacia `main` y describe qué cambia y cómo probarlo. Menciona el Issue
con `Closes #<número>`.

**Nunca hagas push directo a `main`**: la rama está protegida y solo acepta cambios por PR.

## 5. Qué no se sube a Git

- `data/` (cualquier CSV o parquet)
- `mlflow.db`, `mlruns/` y `mlartifacts/`
- `.venv/`, `.env` y `.claude/settings.local.json`

Ya están en `.gitignore`. Revisa `git status` antes de cada commit.

## 6. Uso de IA

Si usas herramientas de IA (asistentes de código, chat, etc.), decláralo:

- En la descripción del PR: qué herramienta usaste y para qué.
- En el reporte de la fase: cada integrante agrega las herramientas que usó en su parte.

No agregues coautorías de IA a los commits; el uso se declara en el PR y en las referencias
del entregable.
