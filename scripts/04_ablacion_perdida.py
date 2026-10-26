"""Paso 4 — E1 + E2(b): U-Net propia; ablación de la pérdida (BCE / Dice / combinada) y de las
conexiones de salto. Comparación en valid, media ± desviación sobre las semillas."""
from dataclasses import replace

import pandas as pd

import _entorno as E
from src.ablacion import correr_ablacion, resumir, por_evento_promedio
from src.configuracion import DIR_RESULTADOS, DIR_FIGURAS
from src.modelo_unet import UNet, contar_parametros, campo_receptivo
from src.utilidades import obtener_dispositivo
from src.visualizacion_ablacion import figura_ablacion

E.titulo("Paso 4/9 — U-Net (E1) y ablación (b): pérdida y conexiones de salto")
modelo = UNet(canales_entrada=2, canales_base=32, profundidad=4)
print(f"U-Net base: {contar_parametros(modelo) / 1e6:.2f} M parámetros | "
      f"campo receptivo {campo_receptivo(UNet(canales_base=8, profundidad=4))} px")

dispositivo = obtener_dispositivo()
base = E.configuracion_base()
variantes = {"bce": replace(base, perdida="bce"), "dice": replace(base, perdida="dice"),
             "combinada": replace(base, perdida="combinada"),
             "sin_saltos": replace(base, perdida="bce", usar_saltos=False)}
semillas = E.PARAMETROS["semillas"]
df = correr_ablacion(variantes, semillas, E.normalizador(), dispositivo, E.DIR_EXPERIMENTOS,
                     tamano_lote=E.PARAMETROS["tamano_lote"], tamano_recorte=E.PARAMETROS["tamano_recorte"],
                     num_workers=E.PARAMETROS["num_workers"])
df.to_csv(DIR_RESULTADOS / "f6b_ablacion_corridas.csv", index=False)
resumen = resumir(df)
resumen.to_csv(DIR_RESULTADOS / "f6b_ablacion_resumen.csv")
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0).loc["valid", "iou"]
figura_ablacion(df, otsu, "Ablación (b) — validación", DIR_FIGURAS / "f6b_ablacion.png")
por_evento_promedio(list(variantes), semillas, E.DIR_EXPERIMENTOS).to_csv(DIR_RESULTADOS / "f6b_ablacion_por_evento.csv")
print(resumen)
