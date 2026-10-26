"""Paso 6 — E3: el modelo elegido SOLO con valid se evalúa una vez en test (umbral 0.5) con todas sus
semillas: IoU, F1, precisión, recall, matriz de confusión, curva IoU-umbral y desglose por evento."""
import pandas as pd

import _entorno as E
from src.cargadores import crear_cargadores
from src.configuracion import DIR_RESULTADOS, DIR_FIGURAS, SEMILLA
from src.evaluacion_final import CARPETAS, elegir_variante, evaluar_semillas, matriz_sumada, por_evento_test
from src.utilidades import obtener_dispositivo
from src.visualizacion_final import figura_curva_umbral
from src.visualizacion_metricas import figura_matriz_confusion

E.titulo("Paso 6/9 — Evaluación en test (E3)")
dispositivo = obtener_dispositivo()
cargadores = crear_cargadores(E.normalizador(), tamano_lote=4, tamano_recorte=256,
                              num_workers=E.PARAMETROS["num_workers"], semilla=SEMILLA)
variante, ranking = elegir_variante(DIR_RESULTADOS / "f6a_ablacion_corridas.csv")
carpeta = CARPETAS[variante]
print(f"Modelo final (elegido en valid): {variante} -> carpetas {carpeta}_s*")
print(ranking.round(3))

resultado = evaluar_semillas(carpeta, E.PARAMETROS["semillas"], cargadores, E.DIR_EXPERIMENTOS, dispositivo)
df = resultado["df"]
df.to_csv(DIR_RESULTADOS / "f7_test_por_semilla.csv", index=False)
metricas = ["iou", "f1", "precision", "recall", "omision", "comision"]
resumen = df.groupby("criterio_umbral")[metricas].agg(["mean", "std"])
resumen.to_csv(DIR_RESULTADOS / "f7_test_resumen.csv")
figura_matriz_confusion(matriz_sumada(resultado["acumuladores_test"]), f"U-Net {variante} — test",
                        DIR_FIGURAS / "f7_matriz_confusion_test.png")
figura_curva_umbral(resultado["curvas_valid"], resultado["curvas_test"], DIR_FIGURAS / "f7_curva_umbral.png")
por_evento_test(resultado["acumuladores_test"]).to_csv(DIR_RESULTADOS / "f7_por_evento_test.csv")
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0).loc["test", "iou"]
print(resumen.round(3))
print(f"IoU test U-Net (0.5) = {df[df['criterio_umbral'] == 'fijo 0.5']['iou'].mean():.3f}  |  Otsu = {otsu:.3f}")
