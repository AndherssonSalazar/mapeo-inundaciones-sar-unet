"""Paso 8 — E6 (bono): fusión temprana de 6 bandas de Sentinel-2 con VV y VH. Tres modos (sar,
optico, sar_optico) con la misma máscara de píxeles; cambio de IoU pareado por semilla."""
import pandas as pd

import _entorno as E
from src.configuracion import DIR_RESULTADOS, DIR_FIGURAS
from src.fusion_optica import MODOS, NormalizadorS2, calcular_estadisticas_s2, diagnostico_s2
from src.lectura import leer_particion
from src.paralelo_e6 import recolectar, resumen_por_modo, deltas_pareados, por_evento
from src.tarea_e6 import correr_tarea
from src.utilidades import obtener_dispositivo
from src.visualizacion_fusion import figura_barras

E.titulo("Paso 8/9 — Fusión con Sentinel-2 (E6, bono)")
dispositivo = obtener_dispositivo()
ruta_s2 = DIR_RESULTADOS / "f10_estadisticas_s2.json"
calcular_estadisticas_s2(leer_particion("train")["id_chip"], ruta_json=ruta_s2)
diagnostico_s2().to_csv(DIR_RESULTADOS / "f10_diagnostico_s2.csv")
normalizador, normalizador_s2 = E.normalizador(), NormalizadorS2.desde_json(ruta_s2)

base = E.configuracion_base(perdida="combinada")
p, semillas = E.PARAMETROS, E.PARAMETROS["semillas"]
for modo in MODOS:
    for semilla in semillas:
        print(f"\n>>> {modo} semilla {semilla}", flush=True)
        correr_tarea(modo, semilla, base, dict(tamano_lote=p["tamano_lote"], tamano_recorte=p["tamano_recorte"],
                                               num_workers=p["num_workers"]),
                     normalizador, normalizador_s2, dispositivo, E.DIR_EXPERIMENTOS)

df = recolectar(semillas, E.DIR_EXPERIMENTOS)
df.to_csv(DIR_RESULTADOS / "f10_corridas.csv", index=False)
resumen = resumen_por_modo(df)
resumen.to_csv(DIR_RESULTADOS / "f10_resumen.csv")
deltas = deltas_pareados(df)
deltas.to_csv(DIR_RESULTADOS / "f10_deltas.csv", index=False)
por_evento(E.DIR_EXPERIMENTOS, semillas, "test").to_csv(DIR_RESULTADOS / "f10_por_evento_test.csv")
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0)["iou"].to_dict()
figura_barras(df, linea_otsu=otsu, ruta_guardado=DIR_FIGURAS / "f10_barras_iou.png")
print(resumen)
print(deltas)
