# Diagnóstico de datos crudos — MiBici 2025-10 a 2026-09

Generado por `notebooks/0.1-lg-diagnostico-raw.ipynb` a partir de `data/raw/` (solo lectura).

## Números clave

- **12 archivos de viajes** + nomenclatura (`nomenclatura_2026_09.csv`, 484 estaciones).
- **4,497,877 viajes** en total, de **32,562 usuarios** distintos.
- Las **8 columnas** tienen el mismo nombre, orden y tipo en todos los meses: `Viaje_Id`, `Usuario_Id`, `Genero`, `Año_de_nacimiento`, `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`.
- Codificación: todos los meses son UTF-8 excepto **2026-01** (Latin-1).
- Fechas: 0 valores con formato distinto de `AAAA-MM-DD HH:MM:SS` y **0 viajes que inician fuera del mes de su archivo**.
- Duplicados: **758 filas duplicadas completas**, todas en 2026-09. Cada `Viaje_Id` repetido corresponde a un duplicado exacto (758 IDs, 1,516 filas); ningún ID se repite entre archivos.

## Por mes

| mes | codificación | filas | nulos Genero | nulos Año_de_nacimiento | duplicadas | inicio fuera de mes | fin en mes siguiente |
|---|---|---|---|---|---|---|---|
| 2025-10 | utf-8 | 420,064 | 4,343 | 33 | 0 | 0 | 24 |
| 2025-11 | utf-8 | 380,328 | 3,894 | 65 | 0 | 0 | 16 |
| 2025-12 | utf-8 | 335,420 | 2,791 | 105 | 0 | 0 | 7 |
| 2026-01 | latin-1 | 357,180 | 2,616 | 90 | 0 | 0 | 21 |
| 2026-02 | utf-8 | 333,756 | 2,400 | 41 | 0 | 0 | 21 |
| 2026-03 | utf-8 | 376,243 | 2,470 | 39 | 0 | 0 | 21 |
| 2026-04 | utf-8 | 351,851 | 2,203 | 22 | 0 | 0 | 42 |
| 2026-05 | utf-8 | 315,895 | 1,704 | 8 | 0 | 0 | 15 |
| 2026-06 | utf-8 | 355,303 | 2,283 | 10 | 0 | 0 | 75 |
| 2026-07 | utf-8 | 388,257 | 2,651 | 61 | 0 | 0 | 34 |
| 2026-08 | utf-8 | 393,524 | 2,332 | 51 | 0 | 0 | 18 |
| 2026-09 | utf-8 | 490,056 | 3,295 | 9 | 758 | 0 | 17 |

"Fin en mes siguiente" son viajes que empiezan el último día del mes y terminan después de la medianoche; es esperado.

## Nulos

Solo hay nulos en dos columnas: `Genero` (32,982, 0.73%) y `Año_de_nacimiento` (534, 0.012%). Las columnas de fechas, IDs y estaciones no tienen nulos.

## Duración de los viajes

| caso | viajes | porcentaje |
|---|---|---|
| negativa | 0 | 0.00 |
| cero | 3,791 | 0.08 |
| mayor a 0 y menor a 1 min | 153,899 | 3.42 |
| menor a 1 min y misma estación | 141,053 | 3.14 |
| mayor a 1 h | 4,356 | 0.10 |
| mayor a 3 h | 510 | 0.01 |
| mayor a 24 h | 13 | 0.00 |

| percentil | minutos |
|---|---|
| p50 | 9.35 |
| p90 | 20.72 |
| p95 | 24.47 |
| p99 | 33.43 |
| p99.9 | 59.18 |
| máximo | 5,920.05 |

## Año de nacimiento y género

- Rango de años: 199 a 2010. Ninguna edad menor a 16 años.
- Años imposibles (antes de 1920): **199** (316 viajes), **1420** (3 viajes). El siguiente año válido es 1939.

| Genero | viajes | porcentaje |
|---|---|---|
| M | 3,241,248 | 72.06 |
| F | 1,223,647 | 27.20 |
| (nulo) | 32,982 | 0.73 |

## Estaciones contra la nomenclatura

- La nomenclatura no tiene IDs duplicados ni nulos; **4 estaciones con capacidad 0**.
- **2 estaciones** aparecen en viajes pero no en la nomenclatura (3,954 viajes afectados, solo en 2026-06, 2026-07; parecen estaciones temporales):

| estacion_id | viajes_como_origen | viajes_como_destino |
|---|---|---|
| 405 | 3 | 3 |
| 407 | 1,957 | 2,120 |

- **15 estaciones** de la nomenclatura no tienen viajes en el periodo; 14 tienen el prefijo `ex-` (dadas de baja). La nomenclatura incluye 14 estaciones `ex-` en total.

| id | name | dpcapacity |
|---|---|---|
| 22 | ex-(GDL-020) Juan Manuel / C.General Coronado | 15 |
| 97 | ex-(ZPN-007) C. Libertad / Nicolás Bravo | 19 |
| 99 | ex-(ZPN-009) C. Del Corral / C. Nicolás Bravo | 0 |
| 109 | ex-(ZPN-019) C. Cabecera Mpal. Juan Manuel / C. | 0 |
| 113 | ex-(ZPN-023) C. Juan Diego / C. Sarcófago | 11 |
| 114 | ex-(ZPN-024) Manuel H. Alatorre / C. Industria | 11 |
| 116 | ex-(ZPN-026) C. 16 Sep. / C. Fray A. de Segovia | 0 |
| 138 | ex-(ZPN-044) | 0 |
| 147 | ex-(TLQ-006) C. Zalatitán / Av. Francisco Silva | 15 |
| 149 | ex-(TLQ-008) C. Juárez / C.Francisco de Miranda | 15 |
| 223 | ex-(GDL-156) C. Crepúsculo / Av. Mariano Otero | 15 |
| 240 | ex-(GDL-173) C. Caballerango / C. Caporales | 19 |
| 250 | ex-GDL-182 Av. Plan de San Luis / Av. Enríque | 15 |
| 310 | ex-(ZPN-063) C. San Felipe Neri / Av. Guadalu | 15 |
| 492 | (GDL-297) Calz. Independencia / Av. República | 19 |

## Propuestas para la limpieza (a decidir en equipo)

1. Leer **2026-01** con `encoding="latin-1"` (o detectar la codificación por archivo).
2. Eliminar las **758 filas duplicadas** exactas.
3. Eliminar duraciones **cero** (3,791). No hay negativas.
4. Viajes de **menos de 1 minuto en la misma estación** (141,053, 3.14%): probablemente bicicletas que se sacan y se regresan de inmediato. Proponemos excluirlos porque no representan demanda real.
5. Corte superior de duración: p99.9 = 59.2 min; solo 510 viajes duran más de 3 h y 13 más de 24 h (máximo 98.7 h). Proponemos un corte en **3 h**. Para pronosticar salidas por estación y hora la duración pesa poco, así que el corte importa más para el EDA.
6. `Año_de_nacimiento` antes de 1920 → nulo (no eliminar el viaje: la salida sí ocurrió).
7. `Genero` nulo → categoría "desconocido".
8. Estaciones 405, 407: no están en la nomenclatura de 2026-09; buscarlas en nomenclaturas anteriores o conservarlas sin coordenadas.
