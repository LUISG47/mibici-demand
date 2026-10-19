"""Limpieza de los viajes de MiBici y versionado de los datos con MLflow.

Uso:  python -m mibici.cleaning
"""

from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, RAW_DATA_DIR, configurar_mlflow

app = typer.Typer()


class DataCleaner:
    """Aplica las reglas de limpieza a un DataFrame de viajes.

    Cada regla es un método pequeño que modifica `self.df` y regresa cuántas filas afectó.
    `run()` las aplica en orden y deja el resultado de cada una en `self.bitacora`.
    """

    # Umbrales (decididos a partir de reports/diagnostico_raw.md)
    DURACION_ATIPICA_MIN = 3 * 60  # más de 3 h: se marca como atípica
    DURACION_MAXIMA_MIN = 24 * 60  # más de 24 h: se elimina
    ANIO_NACIMIENTO_MINIMO = 1920
    FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"

    def __init__(self, viajes: pd.DataFrame, ids_estaciones):
        self.df = viajes.copy()
        self.ids_estaciones = set(ids_estaciones)
        self.bitacora = []

    # ---------- Lectura ----------

    @staticmethod
    def detectar_codificacion(archivo: Path) -> str:
        """'utf-8' si todo el archivo es UTF-8 válido; si no, 'latin-1' (p. ej. 2026-01)."""
        try:
            archivo.read_bytes().decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            return "latin-1"

    @staticmethod
    def leer_csv(archivo: Path) -> pd.DataFrame:
        """Lee un CSV con la codificación que le corresponde."""
        return pd.read_csv(archivo, encoding=DataCleaner.detectar_codificacion(archivo))

    # ---------- Reglas ----------

    def eliminar_duplicados(self) -> int:
        """Quita filas idénticas (758 en 2026-09 según el diagnóstico)."""
        antes = len(self.df)
        self.df = self.df.drop_duplicates()
        return antes - len(self.df)

    def parsear_fechas(self) -> int:
        """Convierte inicio y fin a datetime y agrega duracion_min."""
        for columna in ["Inicio_del_viaje", "Fin_del_viaje"]:
            self.df[columna] = pd.to_datetime(self.df[columna], format=self.FORMATO_FECHA)
        duracion = self.df["Fin_del_viaje"] - self.df["Inicio_del_viaje"]
        self.df["duracion_min"] = duracion.dt.total_seconds() / 60
        return len(self.df)

    def eliminar_duracion_cero(self) -> int:
        """Viajes que terminan en el mismo segundo en que empiezan: no son viajes reales."""
        return self._eliminar(self.df["duracion_min"] == 0)

    def eliminar_cortos_misma_estacion(self) -> int:
        """Menos de 1 minuto y regreso a la misma estación: la bici se sacó y se devolvió."""
        misma_estacion = self.df["Origen_Id"] == self.df["Destino_Id"]
        return self._eliminar((self.df["duracion_min"] < 1) & misma_estacion)

    def eliminar_mas_de_24h(self) -> int:
        """Viajes de más de 24 h: bicicletas no devueltas o errores de registro."""
        return self._eliminar(self.df["duracion_min"] > self.DURACION_MAXIMA_MIN)

    def marcar_duracion_atipica(self) -> int:
        """duracion_atipica = True para viajes de 3 a 24 h. Se conservan."""
        self.df["duracion_atipica"] = self.df["duracion_min"] > self.DURACION_ATIPICA_MIN
        return int(self.df["duracion_atipica"].sum())

    def anio_nacimiento_invalido_a_nulo(self) -> int:
        """Años imposibles (199, 1420) pasan a nulo; el viaje sí ocurrió, no se elimina."""
        invalido = self.df["Año_de_nacimiento"] < self.ANIO_NACIMIENTO_MINIMO
        self.df.loc[invalido, "Año_de_nacimiento"] = None
        return int(invalido.sum())

    def genero_nulo_a_desconocido(self) -> int:
        """Genero vacío pasa a la categoría 'desconocido'."""
        nulo = self.df["Genero"].isna()
        self.df["Genero"] = self.df["Genero"].fillna("desconocido")
        return int(nulo.sum())

    def marcar_estaciones_en_catalogo(self) -> int:
        """estacion_en_catalogo = origen y destino existen en la nomenclatura.

        No se eliminan: las estaciones 405 y 407 sí tuvieron viajes (junio y julio 2026).
        Regresa cuántos viajes quedan fuera del catálogo.
        """
        en_catalogo = self.df["Origen_Id"].isin(self.ids_estaciones) & self.df["Destino_Id"].isin(
            self.ids_estaciones
        )
        self.df["estacion_en_catalogo"] = en_catalogo
        return int((~en_catalogo).sum())

    # ---------- Orquestación ----------

    def run(self) -> pd.DataFrame:
        """Aplica todas las reglas en orden y regresa el DataFrame limpio."""
        reglas = [
            self.eliminar_duplicados,
            self.parsear_fechas,  # las reglas de duración necesitan duracion_min
            self.eliminar_duracion_cero,
            self.eliminar_cortos_misma_estacion,
            self.eliminar_mas_de_24h,
            self.marcar_duracion_atipica,  # después de eliminar >24 h: solo quedan 3 a 24 h
            self.anio_nacimiento_invalido_a_nulo,
            self.genero_nulo_a_desconocido,
            self.marcar_estaciones_en_catalogo,
        ]
        for regla in reglas:
            filas_antes = len(self.df)
            afectadas = regla()
            self.bitacora.append(
                {
                    "regla": regla.__name__,
                    "filas_antes": filas_antes,
                    "filas_despues": len(self.df),
                    "filas_afectadas": afectadas,
                }
            )
            logger.info(f"{regla.__name__}: {afectadas:,} filas afectadas")
        self.df = self.df.reset_index(drop=True)
        return self.df

    def bitacora_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.bitacora)

    def _eliminar(self, mascara: pd.Series) -> int:
        """Quita las filas donde `mascara` es True y regresa cuántas fueron."""
        self.df = self.df[~mascara]
        return int(mascara.sum())


def cargar_viajes(carpeta: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Une todos los CSV mensuales y agrega mes_archivo ('AAAA-MM') para rastrear el origen."""
    partes = []
    for archivo in sorted(carpeta.glob("datos_abiertos_*.csv")):
        df = DataCleaner.leer_csv(archivo)
        df["mes_archivo"] = archivo.stem.removeprefix("datos_abiertos_").replace("_", "-")
        partes.append(df)
    if not partes:
        raise FileNotFoundError(f"No hay CSV de viajes en {carpeta}; corre mibici/dataset.py")
    return pd.concat(partes, ignore_index=True)


def cargar_nomenclatura(carpeta: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Lee la nomenclatura de estaciones más reciente."""
    return DataCleaner.leer_csv(max(carpeta.glob("nomenclatura_*.csv")))


@app.command()
def main(
    salida: Path = INTERIM_DATA_DIR / "viajes_limpios.parquet",
):
    """Limpia data/raw, guarda data/interim/viajes_limpios.parquet y lo versiona en MLflow."""
    configurar_mlflow()
    viajes = cargar_viajes()
    nomenclatura = cargar_nomenclatura()
    meses = sorted(viajes["mes_archivo"].unique())

    with mlflow.start_run(run_name="limpieza"):
        # Dataset crudo
        raw = mlflow.data.from_pandas(viajes, source=str(RAW_DATA_DIR), name="mibici_raw_v1")
        mlflow.log_input(raw, context="raw")

        # Limpieza
        limpiador = DataCleaner(viajes, nomenclatura["id"])
        limpios = limpiador.run()
        limpios.to_parquet(salida, index=False)

        # Dataset limpio
        interim = mlflow.data.from_pandas(limpios, source=str(salida), name="mibici_interim_v1")
        mlflow.log_input(interim, context="interim")

        mlflow.log_params(
            {
                "mes_inicio": meses[0],
                "mes_fin": meses[-1],
                "n_meses": len(meses),
                "meses": ",".join(meses),
                "filas_antes": len(viajes),
                "filas_despues": len(limpios),
            }
        )

        # Bitácora como artefacto CSV
        bitacora = limpiador.bitacora_df()
        with tempfile.TemporaryDirectory() as carpeta_temporal:
            ruta = Path(carpeta_temporal) / "bitacora_limpieza.csv"
            bitacora.to_csv(ruta, index=False)
            mlflow.log_artifact(str(ruta))

    print(bitacora.to_string(index=False))
    logger.success(f"{len(viajes):,} -> {len(limpios):,} filas. Guardado en {salida}")


if __name__ == "__main__":
    app()
