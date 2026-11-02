# CLAUDE.md — mibici-demand

Proyecto de equipo de la materia **MLOps (TC5061)**.

## Proyecto
Pronóstico de **demanda de bicicletas por estación y por hora** (problema de **regresión**),
usando el dataset de datos abiertos de **MiBici Guadalajara**:
https://mibici.net/en/open-data/

## Propuesta de valor
Anticipar qué estaciones van a quedar **vacías o llenas** para poder **rebalancear bicicletas**
a tiempo y mejorar la disponibilidad del servicio.

## Stack
- Python
- pandas
- scikit-learn
- MLflow
- Git / GitHub
- Estructura de proyecto **Cookiecutter Data Science**

## Fase 2 — modelado (entrega interna: 6-nov-2026)
Objetivo: predecir las **salidas de cada estación en la siguiente hora** (horizonte de 1 h)
para el operador que rebalancea bicicletas.
1. Configuración en `params.yaml`, leída con `cargar_parametros()` de `mibici/config.py`
2. Validación walk-forward de ventana creciente (`mibici/modeling/validation.py`),
   prueba mensual de abril a septiembre de 2026; métricas MAE, RMSE y WAPE
3. Dataset `mibici_processed_v3` y Pipeline de scikit-learn ajustado dentro de cada ventana
4. Líneas base, Ridge y HistGradientBoostingRegressor (Poisson) con ajuste de hiperparámetros
5. MLflow: experimento `mibici-fase2-modelado`, run padre por modelo y runs anidados por
   ventana (`mibici/modeling/tracking.py`); Model Registry `mibici-demanda-salidas`

## Fase 1 — completada (entregada el 25-oct-2026)
Limpieza, EDA, preprocesamiento, versionado de datos con MLflow y refactor a módulos con POO.

## Reglas
- Código **simple y legible**.
- **Sin dependencias innecesarias**: usar solo el stack de arriba salvo que haya una razón clara.
- **Comentarios en español.**
- Los **datos NO se suben a Git**: se versionan en MLflow (mantener `data/` en `.gitignore`).
- Excluir también `.venv/` y `mlruns/` de Git.
- Trabajar en ramas `feature/<tema>` y entrar a `main` solo por Pull Request.
- No hacer commits a nombre de otros integrantes del equipo.
