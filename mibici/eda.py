"""
Módulo de análisis exploratorio (EDA) para viajes limpios de MiBici.
"""

from typing import ClassVar

import numpy as np
import pandas as pd


class ViajesEDA:
    """Calcula agregaciones y patrones temporales y espaciales sobre viajes limpios."""

    DIAS_SEMANA: ClassVar[tuple] = (
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    )

    def __init__(self, viajes: pd.DataFrame, nomenclatura: pd.DataFrame | None = None):
        self.viajes = viajes
        self.nomenclatura = nomenclatura

    def obtener_demanda_por_hora(self) -> pd.Series:
        return (
            self.viajes["Inicio_del_viaje"]
            .dt.hour.value_counts()
            .reindex(range(24), fill_value=0)
            .sort_index()
        )

    def obtener_demanda_por_dia_semana(self) -> pd.DataFrame:
        demanda_total = (
            self.viajes["Inicio_del_viaje"]
            .dt.dayofweek.value_counts()
            .reindex(range(7), fill_value=0)
            .sort_index()
        )
        fechas_observadas = self.viajes["Inicio_del_viaje"].dt.normalize().drop_duplicates()
        dias_observados = (
            fechas_observadas.dt.dayofweek.value_counts()
            .reindex(range(7), fill_value=0)
            .sort_index()
        )

        return pd.DataFrame(
            {
                "dia_semana": self.DIAS_SEMANA,
                "total_viajes": demanda_total.values,
                "dias_observados": dias_observados.values,
                "promedio_viajes_dia": demanda_total.div(
                    dias_observados.replace(0, np.nan)
                ).values,
            }
        )

    def obtener_matriz_dia_hora(self) -> pd.DataFrame:
        demanda_dia_hora = (
            self.viajes.groupby(
                [
                    self.viajes["Inicio_del_viaje"].dt.dayofweek,
                    self.viajes["Inicio_del_viaje"].dt.hour,
                ]
            )
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )
        fechas_observadas = self.viajes["Inicio_del_viaje"].dt.normalize().drop_duplicates()
        dias_observados = fechas_observadas.dt.dayofweek.value_counts().reindex(
            range(7), fill_value=0
        )

        matriz_promedio = demanda_dia_hora.div(dias_observados.replace(0, np.nan), axis=0)
        matriz_promedio.index = list(self.DIAS_SEMANA)
        return matriz_promedio

    def obtener_top_estaciones(self, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        salidas = (
            self.viajes["Origen_Id"]
            .value_counts()
            .head(top_n)
            .rename_axis("id")
            .reset_index(name="total_salidas")
        )
        llegadas = (
            self.viajes["Destino_Id"]
            .value_counts()
            .head(top_n)
            .rename_axis("id")
            .reset_index(name="total_llegadas")
        )

        if self.nomenclatura is not None:
            cat = self.nomenclatura[["id", "name"]]
            salidas = salidas.merge(cat, on="id", how="left")[["id", "name", "total_salidas"]]
            llegadas = llegadas.merge(cat, on="id", how="left")[["id", "name", "total_llegadas"]]

        return salidas, llegadas

    def obtener_balance_estaciones(self) -> pd.DataFrame:
        salidas = self.viajes["Origen_Id"].value_counts().rename("salidas")
        llegadas = self.viajes["Destino_Id"].value_counts().rename("llegadas")

        balance = pd.concat([salidas, llegadas], axis=1).fillna(0).astype(int)
        balance.index.name = "id"
        balance = balance.reset_index()
        balance["balance_neto"] = balance["llegadas"] - balance["salidas"]

        if self.nomenclatura is not None:
            cat = self.nomenclatura[["id", "name", "dpcapacity"]]
            balance = balance.merge(cat, on="id", how="left")

        return balance

    def validar_catalogo_estaciones(self) -> bool:
        """
        Verifica que el catálogo no tenga IDs nulos ni duplicados.
        """
        if self.nomenclatura is None:
            raise ValueError("No se proporcionó nomenclatura al inicializar ViajesEDA.")

        assert self.nomenclatura["id"].notna().all(), (
            "El catálogo contiene identificadores de estación nulos."
        )
        assert self.nomenclatura["id"].is_unique, (
            "El catálogo contiene identificadores de estación duplicados."
        )
        return True

    def obtener_duraciones_visibles(self, quantil: float = 0.99) -> tuple[pd.Series, float, int]:
        """Filtra duraciones hasta el cuantil especificado para descartar outliers visuales."""
        limite = self.viajes["duracion_min"].quantile(quantil)
        visibles = self.viajes.loc[self.viajes["duracion_min"] <= limite, "duracion_min"]
        fuera_rango = len(self.viajes) - len(visibles)
        return visibles, limite, fuera_rango

    def obtener_distribucion_genero(self) -> pd.DataFrame:
        """Calcula el total y porcentaje de viajes por género."""
        dist = self.viajes["Genero"].value_counts().reset_index()
        dist.columns = ["genero", "total_viajes"]
        dist["porcentaje"] = (dist["total_viajes"] / len(self.viajes)) * 100
        return dist

    def obtener_edades_estimadas(self) -> pd.Series:
        """Calcula la edad estimada basándose en el año de inicio del viaje."""
        edades = self.viajes["Inicio_del_viaje"].dt.year - self.viajes["Año_de_nacimiento"]
        return edades

    def obtener_actividad_geografica(self) -> pd.DataFrame:
        """Cruza los movimientos por estación con las coordenadas del catálogo."""
        if self.nomenclatura is None:
            raise ValueError("No se proporcionó nomenclatura al inicializar ViajesEDA.")

        salidas = self.viajes["Origen_Id"].value_counts()
        llegadas = self.viajes["Destino_Id"].value_counts()
        movimientos = salidas.add(llegadas, fill_value=0).rename("total_movimientos")

        geo = self.nomenclatura[["id", "latitude", "longitude"]].copy()
        geo = geo.merge(movimientos, left_on="id", right_index=True, how="inner")

        # Filtrar coordenadas válidas (que no sean nulas ni 0)
        geo = geo.dropna(subset=["latitude", "longitude"])
        geo = geo[(geo["latitude"] != 0) & (geo["longitude"] != 0)]
        return geo

    def obtener_metricas_mlflow(self) -> dict:
        """Consolida las métricas descriptivas escalares para registrar en MLflow."""
        edades = self.obtener_edades_estimadas()
        edades_validas = edades.dropna()
        balance = self.obtener_balance_estaciones()

        return {
            "total_viajes": len(self.viajes),
            "total_estaciones": len(balance),
            "duracion_media_min": float(self.viajes["duracion_min"].mean()),
            "duracion_mediana_min": float(self.viajes["duracion_min"].median()),
            "edad_media_estimada": float(edades_validas.mean()),
            "viajes_duracion_atipica": int(
                self.viajes.get("duracion_atipica", pd.Series([0])).sum()
            ),
            "viajes_fuera_catalogo": int(
                (~self.viajes.get("estacion_en_catalogo", pd.Series([True]))).sum()
            ),
            "viajes_sin_edad": int(edades.isna().sum()),
        }
