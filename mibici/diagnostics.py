"""
Módulo de diagnóstico para datos crudos de MiBici.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DiagnosticConfig:
    min_year_valid: int = 1920
    max_duration_hours: int = 3
    date_format: str = "%Y-%m-%d %H:%M:%S"


class RawDataDiagnoser:
    """
    Diagnostica la integridad, tipos y calidad de los datos crudos.
    """

    def __init__(self, config: DiagnosticConfig | None = None):
        self.config = config or DiagnosticConfig()

    @staticmethod
    def extract_month_from_path(path: Path) -> str:
        """
        Extrae el mes 'YYYY-MM' del nombre de archivo 'datos_abiertos_YYYY_MM.csv'.
        """
        return path.stem.removeprefix("datos_abiertos_").replace("_", "-")

    @staticmethod
    def detect_encoding(path: Path) -> str:
        """
        Determina si un archivo es UTF-8 o requiere Latin-1.
        """
        try:
            path.read_bytes().decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    @staticmethod
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        """
        Genera una tabla Markdown pura a partir de un DataFrame sin dependencias externas.
        """

        def _fmt(val: Any) -> str:
            if isinstance(val, bool):
                return "sí" if val else "no"
            if isinstance(val, int):
                return f"{val:,}"
            if isinstance(val, float):
                return f"{val:,.2f}"
            return str(val)

        header = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "---|" * len(df.columns)
        rows = [
            "| " + " | ".join(_fmt(v) for v in row) + " |"
            for row in df.astype(object).itertuples(index=False)
        ]
        return "\n".join([header, sep, *rows])

    def compute_duration_metrics(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Calcula conteos y percentiles de la duración de viajes.
        """
        duration = (df["Fin_del_viaje"] - df["Inicio_del_viaje"]).dt.total_seconds() / 60
        same_station = df["Origen_Id"] == df["Destino_Id"]

        counts = pd.Series(
            {
                "negativa": (duration < 0).sum(),
                "cero": (duration == 0).sum(),
                "mayor a 0 y menor a 1 min": ((duration > 0) & (duration < 1)).sum(),
                "menor a 1 min y misma estación": (
                    (duration > 0) & (duration < 1) & same_station
                ).sum(),
                "mayor a 1 h": (duration > 60).sum(),
                "mayor a 3 h": (duration > 180).sum(),
                "mayor a 24 h": (duration > 1440).sum(),
            },
            name="viajes",
        ).to_frame()
        total_rows = max(len(df), 1)
        counts["porcentaje"] = (counts["viajes"] / total_rows * 100).round(3)

        percentiles = (
            duration.quantile([0.5, 0.9, 0.95, 0.99, 0.999])
            .rename(lambda p: f"p{p * 100:g}")
            .to_frame("minutos")
        )
        percentiles.loc["máximo"] = duration.max() if not duration.empty else 0.0

        return {"counts": counts, "percentiles": percentiles.round(2)}

    def check_station_consistency(
        self, df_trips: pd.DataFrame, df_nomenclature: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Contrasta IDs de origen/destino contra la nomenclatura disponible.
        """
        valid_ids = set(df_nomenclature["id"].dropna().unique())
        missing_origin = df_trips.loc[~df_trips["Origen_Id"].isin(valid_ids), "Origen_Id"]
        missing_dest = df_trips.loc[~df_trips["Destino_Id"].isin(valid_ids), "Destino_Id"]

        missing_summary = (
            pd.DataFrame(
                {
                    "viajes_como_origen": missing_origin.value_counts(),
                    "viajes_como_destino": missing_dest.value_counts(),
                }
            )
            .fillna(0)
            .astype(int)
            .sort_index()
        )
        missing_summary.index.name = "estacion_id"

        used_ids = set(df_trips["Origen_Id"].dropna().unique()) | set(
            df_trips["Destino_Id"].dropna().unique()
        )
        unused_stations = df_nomenclature[~df_nomenclature["id"].isin(used_ids)]

        return {
            "missing_stations": missing_summary,
            "unused_stations": unused_stations,
        }
