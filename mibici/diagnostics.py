"""
Módulo de diagnóstico para datos crudos de MiBici.
"""

from pathlib import Path
from typing import Any

import pandas as pd


class RawDataDiagnoser:
    """
    Diagnostica la integridad, tipos y calidad de los datos crudos de MiBici.
    """

    DURACION_CORTA_MIN = 1.0
    DURACION_ATIPICA_MIN = 3 * 60  # > 3 h
    DURACION_MAXIMA_MIN = 24 * 60  # > 24 h
    ANIO_NACIMIENTO_MINIMO = 1920

    def __init__(
        self,
        viajes: pd.DataFrame,
        nomenclatura: pd.DataFrame | None = None,
    ):
        self.df = viajes
        self.nomenclatura = nomenclatura
        self.resultados: dict[str, Any] = {}

    @staticmethod
    def extraer_mes_de_ruta(ruta: Path) -> str:
        """Extrae el mes 'YYYY-MM' del nombre de archivo 'datos_abiertos_YYYY_MM.csv'."""
        return ruta.stem.removeprefix("datos_abiertos_").replace("_", "-")

    @staticmethod
    def detectar_codificacion(ruta: Path) -> str:
        """Determina si un archivo es UTF-8 o requiere Latin-1."""
        try:
            ruta.read_bytes().decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    @staticmethod
    def dataframe_a_markdown(df: pd.DataFrame) -> str:
        """Genera una tabla Markdown pura a partir de un DataFrame sin dependencias externas."""

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
        filas = [
            "| " + " | ".join(_fmt(v) for v in fila) + " |"
            for fila in df.astype(object).itertuples(index=False)
        ]
        return "\n".join([header, sep, *filas])

    def diagnosticar_valores_nulos(self) -> pd.DataFrame:
        """Calcula conteos y porcentajes de valores nulos por columna."""
        total = max(len(self.df), 1)
        nulos = self.df.isna().sum()
        resumen = pd.DataFrame(
            {
                "columna": nulos.index,
                "nulos": nulos.values,
                "porcentaje": (nulos.values / total * 100).round(2),
            }
        ).sort_values("nulos", ascending=False)
        self.resultados["nulos"] = resumen
        return resumen

    def diagnosticar_duplicados(self) -> int:
        """Conteo de filas idénticas en el conjunto de datos."""
        conteo = int(self.df.duplicated().sum())
        self.resultados["duplicados"] = conteo
        return conteo

    def diagnosticar_rango_fechas(self) -> dict[str, Any]:
        """Obtiene la fecha mínima, máxima y meses cubiertos."""
        inicio = pd.to_datetime(self.df["Inicio_del_viaje"])
        fin = pd.to_datetime(self.df["Fin_del_viaje"])
        resumen = {
            "fecha_minima": inicio.min(),
            "fecha_maxima": fin.max(),
            "total_dias": (fin.max() - inicio.min()).days if not inicio.empty else 0,
        }
        self.resultados["rango_fechas"] = resumen
        return resumen

    def diagnosticar_edad_y_genero(self) -> dict[str, Any]:
        """Audita años de nacimiento inválidos y distribución por género."""
        conteo_genero = self.df["Genero"].value_counts(dropna=False)

        anios = self.df["Año_de_nacimiento"].dropna()
        invalidos = (anios < self.ANIO_NACIMIENTO_MINIMO).sum()

        resumen = {
            "distribucion_genero": conteo_genero,
            "anios_invalidos": int(invalidos),
            "anio_minimo": float(anios.min()) if not anios.empty else None,
            "anio_maximo": float(anios.max()) if not anios.empty else None,
        }
        self.resultados["demograficos"] = resumen
        return resumen

    def computar_metricas_de_duracion(
        self, df: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
        """Calcula conteos y percentiles de la duración de viajes."""
        datos = self.df if df is None else df
        inicio = pd.to_datetime(datos["Inicio_del_viaje"])
        fin = pd.to_datetime(datos["Fin_del_viaje"])
        duracion = (fin - inicio).dt.total_seconds() / 60
        misma_estacion = datos["Origen_Id"] == datos["Destino_Id"]

        conteos = pd.Series(
            {
                "negativa": (duracion < 0).sum(),
                "cero": (duracion == 0).sum(),
                "mayor a 0 y menor a 1 min": (
                    (duracion > 0) & (duracion < self.DURACION_CORTA_MIN)
                ).sum(),
                "menor a 1 min y misma estación": (
                    (duracion < self.DURACION_CORTA_MIN) & misma_estacion
                ).sum(),
                "mayor a 1 h": (duracion > 60).sum(),
                "mayor a 3 h": (duracion > self.DURACION_ATIPICA_MIN).sum(),
                "mayor a 24 h": (duracion > self.DURACION_MAXIMA_MIN).sum(),
            },
            name="viajes",
        ).to_frame()

        total_filas = max(len(datos), 1)
        conteos["porcentaje"] = (conteos["viajes"] / total_filas * 100).round(3)

        percentiles = (
            duracion.quantile([0.5, 0.9, 0.95, 0.99, 0.999])
            .rename(lambda p: f"p{p * 100:g}")
            .to_frame("minutos")
        )
        percentiles.loc["máximo"] = duracion.max() if not duracion.empty else 0.0

        resumen = {"counts": conteos, "percentiles": percentiles.round(2)}
        self.resultados["duracion"] = resumen
        return resumen

    def revisar_consistencia_de_estaciones(
        self,
        df_viajes: pd.DataFrame | None = None,
        df_nomenclatura: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Contrasta IDs de origen/destino contra la nomenclatura disponible."""
        viajes = self.df if df_viajes is None else df_viajes
        nomenclatura = self.nomenclatura if df_nomenclatura is None else df_nomenclatura

        if nomenclatura is None:
            return {}

        ids_validos = set(nomenclatura["id"].dropna().unique())
        origen_invalido = viajes.loc[~viajes["Origen_Id"].isin(ids_validos), "Origen_Id"]
        destino_invalido = viajes.loc[~viajes["Destino_Id"].isin(ids_validos), "Destino_Id"]

        resumen_faltantes = (
            pd.DataFrame(
                {
                    "viajes_como_origen": origen_invalido.value_counts(),
                    "viajes_como_destino": destino_invalido.value_counts(),
                }
            )
            .fillna(0)
            .astype(int)
            .sort_index()
        )
        resumen_faltantes.index.name = "estacion_id"

        ids_usados = set(viajes["Origen_Id"].dropna().unique()) | set(
            viajes["Destino_Id"].dropna().unique()
        )
        estaciones_no_usadas = nomenclatura[~nomenclatura["id"].isin(ids_usados)]

        resumen = {
            "missing_stations": resumen_faltantes,
            "unused_stations": estaciones_no_usadas,
        }
        self.resultados["estaciones"] = resumen
        return resumen

    def ejecutar(self) -> dict[str, Any]:
        """
        Ejecuta toda la suite de diagnóstico y devuelve el resumen completo.
        """
        self.diagnosticar_valores_nulos()
        self.diagnosticar_duplicados()
        self.diagnosticar_rango_fechas()
        self.diagnosticar_edad_y_genero()
        self.computar_metricas_de_duracion()
        if self.nomenclatura is not None:
            self.revisar_consistencia_de_estaciones()
        return self.resultados
