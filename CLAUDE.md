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

## Fase 1 — entrega: 25-oct-2026
1. Limpieza de datos
2. Análisis exploratorio (EDA)
3. Preprocesamiento
4. Versionado de datos con MLflow
5. Refactor del código a módulos con POO (programación orientada a objetos)

## Reglas
- Código **simple y legible**.
- **Sin dependencias innecesarias**: usar solo el stack de arriba salvo que haya una razón clara.
- **Comentarios en español.**
- Los **datos NO se suben a Git**: se versionan en MLflow (mantener `data/` en `.gitignore`).
- Excluir también `.venv/` y `mlruns/` de Git.
- Trabajar en ramas `feature/<tema>` y entrar a `main` solo por Pull Request.
- No hacer commits a nombre de otros integrantes del equipo.
