"""Paso 9 — Tabla final con las cifras principales del informe (IoU de la clase agua)."""
import pandas as pd

import _entorno as E
from src.configuracion import DIR_RESULTADOS

E.titulo("Paso 9/9 — Resumen de resultados")
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0)["iou"]
regiones = pd.read_csv(DIR_RESULTADOS / "f8_brecha_regiones.csv").set_index("region")
fusion = pd.read_csv(DIR_RESULTADOS / "f10_corridas.csv")
fusion = fusion[fusion["modo"] == "sar_optico"].groupby("particion")["iou"].agg(["mean", "std"])

def texto(media, desv):
    return f"{media:.3f}" + ("" if pd.isna(desv) else f" ± {desv:.3f}")

filas = []
for region in ["test", "bolivia", "lima"]:
    if region not in regiones.index:
        continue
    fila = regiones.loc[region]
    filas.append({"evaluación": region,
                  "Otsu": f"{fila['iou_otsu']:.3f}",
                  "U-Net (VV, VH)": texto(fila["iou_unet"], fila.get("iou_unet_desv")),
                  "Fusión SAR + S2": texto(*fusion.loc[region]) if region in fusion.index else "—"})
tabla = pd.DataFrame(filas)
tabla.to_csv(DIR_RESULTADOS / "resumen_final.csv", index=False)
print(tabla.to_string(index=False))
print(f"\nTodo listo. Resultados en {DIR_RESULTADOS} y figuras en {DIR_RESULTADOS.parent / 'figuras'}")
