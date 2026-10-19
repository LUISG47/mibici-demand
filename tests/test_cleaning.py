import pandas as pd

from mibici.cleaning import DataCleaner


def viajes_ejemplo():
    """Cinco viajes pequeños; las estaciones 1 y 2 están en el catálogo, la 407 no."""
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 1, 2, 3, 4],
            "Usuario_Id": [10, 10, 20, 30, 40],
            "Genero": ["M", "M", None, "F", "M"],
            "Año_de_nacimiento": [1990.0, 1990.0, 199.0, 1985.0, 2000.0],
            "Inicio_del_viaje": [
                "2026-01-01 08:00:00",  # duplicado exacto de la siguiente fila
                "2026-01-01 08:00:00",
                "2026-01-01 09:00:00",  # 30 segundos y misma estación
                "2026-01-01 10:00:00",  # 5 horas: atípico
                "2026-01-01 11:00:00",  # 2 días: se elimina
            ],
            "Fin_del_viaje": [
                "2026-01-01 08:15:00",
                "2026-01-01 08:15:00",
                "2026-01-01 09:00:30",
                "2026-01-01 15:00:00",
                "2026-01-03 11:00:00",
            ],
            "Origen_Id": [1, 1, 2, 1, 2],
            "Destino_Id": [2, 2, 2, 407, 1],
        }
    )


def test_run_elimina_duplicados_cortos_y_mas_de_24h():
    limpiador = DataCleaner(viajes_ejemplo(), ids_estaciones=[1, 2])
    limpios = limpiador.run()

    # Solo sobreviven el viaje 1 (sin su duplicado) y el viaje 3 (atípico)
    assert limpios["Viaje_Id"].tolist() == [1, 3]
    assert limpios["duracion_atipica"].tolist() == [False, True]
    assert limpios["duracion_min"].tolist() == [15.0, 300.0]


def test_bitacora_registra_filas_antes_y_despues():
    limpiador = DataCleaner(viajes_ejemplo(), ids_estaciones=[1, 2])
    limpiador.run()
    bitacora = limpiador.bitacora_df().set_index("regla")

    assert bitacora.loc["eliminar_duplicados", "filas_antes"] == 5
    assert bitacora.loc["eliminar_duplicados", "filas_despues"] == 4
    assert bitacora.loc["eliminar_cortos_misma_estacion", "filas_afectadas"] == 1
    assert bitacora.loc["eliminar_mas_de_24h", "filas_afectadas"] == 1


def test_anio_invalido_y_genero_nulo_no_eliminan_el_viaje():
    limpiador = DataCleaner(viajes_ejemplo(), ids_estaciones=[1, 2])

    assert limpiador.anio_nacimiento_invalido_a_nulo() == 1
    assert limpiador.genero_nulo_a_desconocido() == 1
    assert len(limpiador.df) == 5
    assert pd.isna(limpiador.df.loc[2, "Año_de_nacimiento"])
    assert limpiador.df.loc[2, "Genero"] == "desconocido"


def test_estacion_fuera_de_catalogo_se_marca_sin_eliminar():
    limpiador = DataCleaner(viajes_ejemplo(), ids_estaciones=[1, 2])

    assert limpiador.marcar_estaciones_en_catalogo() == 1
    assert limpiador.df["estacion_en_catalogo"].tolist() == [True, True, True, False, True]


def test_leer_csv_detecta_latin1(tmp_path):
    archivo = tmp_path / "latin1.csv"
    archivo.write_bytes("Año_de_nacimiento\n1990\n".encode("latin-1"))

    assert DataCleaner.detectar_codificacion(archivo) == "latin-1"
    assert DataCleaner.leer_csv(archivo).columns.tolist() == ["Año_de_nacimiento"]
