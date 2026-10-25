"""Módulo de visualización y generación de gráficas para MiBici."""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from mibici.config import FIGURES_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def graficar_distribucion_duracion_viajes(
    durations_minutes: pd.Series,
    max_cutoff: float = 60.0,
    title: str = "Distribución de Duración de Viajes (min)",
) -> plt.Figure:
    """Genera un histograma con estimación de densidad (KDE) para duraciones válidas."""
    filtered = durations_minutes[(durations_minutes > 0) & (durations_minutes <= max_cutoff)]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered, bins=30, kde=True, ax=ax, color="#1f77b4")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Duración (minutos)")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


class EDAPlots:
    """Clase que encapsula las gráficas para el Análisis Exploratorio de Datos (EDA)."""

    @staticmethod
    def graficar_demanda_por_hora(
        demanda_por_hora: pd.Series,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        """Genera gráfico de barras con la demanda de viajes por hora del día."""
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            x=demanda_por_hora.index,
            y=demanda_por_hora.values,
            color="steelblue",
            ax=ax,
        )
        ax.set_title("Demanda de viajes MiBici por hora del día")
        ax.set_xlabel("Hora de inicio del viaje")
        ax.set_ylabel("Número de viajes")
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_demanda_por_dia_semana(
        resumen_demanda_dia: pd.DataFrame,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        """Genera gráfico de barras con el promedio diario de viajes por día de la semana."""
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(
            data=resumen_demanda_dia,
            x="dia_semana",
            y="promedio_viajes_dia",
            color="steelblue",
            ax=ax,
        )
        ax.set_title("Demanda promedio diaria de viajes MiBici por día de la semana")
        ax.set_xlabel("Día de la semana")
        ax.set_ylabel("Promedio de viajes por día observado")
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_mapa_calor_dia_hora(
        matriz_dia_hora: pd.DataFrame,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        """Genera mapa de calor de demanda promedio por día y hora."""
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            matriz_dia_hora,
            cmap="YlOrRd",
            annot=False,
            cbar_kws={"label": "Promedio de viajes por día y hora"},
            ax=ax,
        )
        ax.set_title("Demanda promedio de MiBici por día y hora")
        ax.set_xlabel("Hora de inicio del viaje")
        ax.set_ylabel("Día de la semana")
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_top_estaciones(
        top_salidas: pd.DataFrame,
        top_llegadas: pd.DataFrame,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        """Genera comparativa en dos paneles de las estaciones con más salidas y llegadas."""
        fig, axes = plt.subplots(1, 2, figsize=(17, 6))

        sns.barplot(
            data=top_salidas,
            x="total_salidas",
            y="name",
            color="steelblue",
            ax=axes[0],
        )
        axes[0].set_title("Top 10 estaciones con más salidas")
        axes[0].set_xlabel("Número de salidas")
        axes[0].set_ylabel("Estación")
        axes[0].grid(axis="x", alpha=0.3)

        sns.barplot(
            data=top_llegadas,
            x="total_llegadas",
            y="name",
            color="seagreen",
            ax=axes[1],
        )
        axes[1].set_title("Top 10 estaciones con más llegadas")
        axes[1].set_xlabel("Número de llegadas")
        axes[1].set_ylabel("Estación")
        axes[1].grid(axis="x", alpha=0.3)

        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_balance_estaciones(
        balance_destacado: pd.DataFrame,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        """Genera gráfico de barras horizontales divergente del balance neto por estación."""
        colores = [
            "indianred" if balance < 0 else "seagreen"
            for balance in balance_destacado["balance_neto"]
        ]
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.barh(
            balance_destacado["etiqueta"],
            balance_destacado["balance_neto"],
            color=colores,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title("Top estaciones por balance neto de viajes (llegadas - salidas)")
        ax.set_xlabel("Balance neto de viajes")
        ax.set_ylabel("Estación")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_distribucion_duracion(
        duraciones_visibles: pd.Series, mediana: float, ruta_guardado: Path | None = None
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(duraciones_visibles, bins=50, color="steelblue", ax=axes[0])
        axes[0].axvline(mediana, color="indianred", linestyle="--", label="Mediana")
        axes[0].set_title("Distribución de duración de los viajes")
        axes[0].set_xlabel("Duración del viaje (minutos)")
        axes[0].set_ylabel("Número de viajes")
        axes[0].legend()

        axes[1].boxplot(
            duraciones_visibles.dropna(),
            orientation="horizontal",
            patch_artist=True,
            boxprops={"facecolor": "steelblue"},
        )
        axes[1].set_title("Dispersión de duración de los viajes")
        axes[1].set_xlabel("Duración del viaje (minutos)")
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_distribucion_genero(
        distribucion_genero: pd.DataFrame,
        ruta_guardado: Path | None = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

        sns.barplot(
            data=distribucion_genero,
            x="genero",
            y="total_viajes",
            color="steelblue",
            ax=ax,
        )

        max_viajes = distribucion_genero["total_viajes"].max()
        offset = max_viajes * 0.02

        for i, fila in distribucion_genero.reset_index(drop=True).iterrows():
            ax.text(
                i,
                fila["total_viajes"] + offset,
                f"{fila['porcentaje']:.2f}%",
                ha="center",
                va="bottom",
            )

        ax.set_title("Distribución de viajes MiBici por género registrado")
        ax.set_xlabel("Género registrado")
        ax.set_ylabel("Número de viajes")
        ax.set_ylim(0, max_viajes * 1.15)
        ax.ticklabel_format(style="plain", axis="y")
        ax.grid(axis="y", alpha=0.3)

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")

        return fig

    @staticmethod
    def graficar_distribucion_edad(
        edades_validas: pd.Series, ruta_guardado: Path | None = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.histplot(
            edades_validas, bins=range(15, 91, 5), color="steelblue", edgecolor="white", ax=ax
        )
        ax.axvline(
            edades_validas.median(),
            color="indianred",
            linestyle="--",
            label=f"Mediana: {edades_validas.median():.0f} años",
        )
        ax.set_title("Distribución de viajes MiBici por edad estimada")
        ax.set_xlabel("Edad estimada al momento del viaje (años)")
        ax.set_ylabel("Número de viajes")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig

    @staticmethod
    def graficar_mapa_estaciones(
        estaciones_mapa: pd.DataFrame, ruta_guardado: Path | None = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.scatterplot(
            data=estaciones_mapa,
            x="longitude",
            y="latitude",
            size="total_movimientos",
            sizes=(20, 350),
            color="steelblue",
            alpha=0.65,
            legend="brief",
            ax=ax,
        )
        ax.set_title("Distribución geográfica y actividad de estaciones MiBici")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        if ruta_guardado:
            ruta_guardado.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
        return fig


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "viajes_limpios.parquet",
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
    fig_dur = graficar_distribucion_duracion_viajes(df["duracion_min"])
    fig_dur.savefig(output_dir / "distribucion_duracion_viajes.png")
    logger.success(f"Figuras guardadas exitosamente en {output_dir}")


if __name__ == "__main__":
    app()
