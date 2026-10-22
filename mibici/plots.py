"""Módulo de visualización y generación de gráficas para MiBici."""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from mibici.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def plot_missing_values(
    df: pd.DataFrame, title: str = "Porcentaje de Valores Nulos por Columna"
) -> plt.Figure:
    """
    Genera un gráfico de barras con el porcentaje de nulos por columna
    """
    null_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=null_pct.values,
        y=null_pct.index,
        hue=null_pct.index,
        legend=False,
        ax=ax,
        palette="mako",
        orient="h",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Nulos (%)")
    ax.set_ylabel("Columnas")
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def plot_trip_duration_distribution(
    durations_minutes: pd.Series,
    max_cutoff: float = 60.0,
    title: str = "Distribución de Duración de Viajes (min)",
) -> plt.Figure:
    """
    Genera un histograma con estimación de densidad (KDE) para duraciones válidas.
    """
    filtered = durations_minutes[(durations_minutes > 0) & (durations_minutes <= max_cutoff)]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered, bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Duración (minutos)")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def plot_monthly_trip_counts(
    monthly_counts: pd.Series, title: str = "Volumen Total de Viajes por Mes"
) -> plt.Figure:
    """
    Genera un gráfico de barras con el número total de viajes por mes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_counts.plot(kind="bar", ax=ax, color="#2b5c8f")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Número de viajes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def plot_gender_distribution(
    gender_counts: pd.Series, title: str = "Distribución de Viajes por Género"
) -> plt.Figure:
    """
    Genera un gráfico de barras por categoría de género incluyendo nulos.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    gender_counts.plot(kind="bar", ax=ax, color="#488f31")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Género")
    ax.set_ylabel("Total de viajes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "mibici_processed.parquet",
    output_dir: Path = FIGURES_DIR,
):
    """
    Punto de entrada CLI para generar y guardar figuras base.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generando figuras diagnósticas desde {input_path}...")

    if not input_path.exists():
        logger.warning(f"El archivo {input_path} no existe aún. Saltando generación de plots.")
        return

    df = pd.read_parquet(input_path)
    fig_missing = plot_missing_values(df)
    fig_missing.savefig(output_dir / "missing_values.png")
    logger.success(f"Figuras guardadas exitosamente en {output_dir}")


if __name__ == "__main__":
    app()
