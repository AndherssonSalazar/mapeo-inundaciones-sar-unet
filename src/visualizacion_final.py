"""
visualizacion_final.py
Mapa de errores por píxel y curva IoU-umbral (fase 7; también se reutiliza en el análisis de fallos).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# 0 ignorado, 1 VN, 2 VP, 3 FP (agua inventada), 4 FN (agua perdida)
MAPA_ERRORES = ListedColormap(["#ffffff", "#e8e3d6", "#1f5fbf", "#d62728", "#ff9f1c"])
LEYENDA = [Patch(color=c, label=t) for c, t in zip(MAPA_ERRORES.colors[1:], ["VN", "VP", "FP (comisión)", "FN (omisión)"])]


def mapa_errores(etiqueta, probabilidad, umbral=0.5):
    prediccion = probabilidad > umbral
    mapa = np.zeros(etiqueta.shape, dtype=np.int8)
    validos = etiqueta != -1
    mapa[validos & ~prediccion & (etiqueta == 0)] = 1
    mapa[validos & prediccion & (etiqueta == 1)] = 2
    mapa[validos & prediccion & (etiqueta == 0)] = 3
    mapa[validos & ~prediccion & (etiqueta == 1)] = 4
    return mapa


def figura_errores(ejemplos, umbral=0.5, ruta_guardado=None):
    """ejemplos: [(id_chip, vh_normalizado, etiqueta, probabilidad), ...] -> VH | P(agua) | mapa de errores."""
    figura, ejes = plt.subplots(len(ejemplos), 3, figsize=(10, 3.3 * len(ejemplos)))
    ejes = np.atleast_2d(ejes)
    for fila, (id_chip, vh, etiqueta, probabilidad) in enumerate(ejemplos):
        ejes[fila, 0].imshow(vh, cmap="gray", vmin=-2.5, vmax=2.5); ejes[fila, 0].set_title(f"{id_chip} — VH", fontsize=9)
        ejes[fila, 1].imshow(probabilidad, cmap="Blues", vmin=0, vmax=1); ejes[fila, 1].set_title("P(agua)", fontsize=9)
        ejes[fila, 2].imshow(mapa_errores(etiqueta, probabilidad, umbral), cmap=MAPA_ERRORES, vmin=0, vmax=4,
                             interpolation="nearest")
        ejes[fila, 2].set_title("Errores", fontsize=9)
        for eje in ejes[fila]:
            eje.axis("off")
    figura.legend(handles=LEYENDA, loc="lower center", ncol=4, frameon=False)
    figura.tight_layout(rect=(0, 0.03, 1, 1))
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura


def figura_curva_umbral(curvas_valid, curvas_test, ruta_guardado=None):
    figura, eje = plt.subplots(figsize=(6.5, 4))
    for semilla in curvas_valid:
        eje.plot(curvas_valid[semilla]["umbral"], curvas_valid[semilla]["iou"], color="#7a9cc6", alpha=0.8,
                 label="valid" if semilla == list(curvas_valid)[0] else None)
        eje.plot(curvas_test[semilla]["umbral"], curvas_test[semilla]["iou"], color="#1f5fbf",
                 label="test" if semilla == list(curvas_valid)[0] else None)
    eje.axvline(0.5, ls="--", color="gray", label="umbral 0.5")
    eje.set_xlabel("Umbral sobre P(agua)"); eje.set_ylabel("IoU agua"); eje.set_title("IoU vs. umbral (una línea por semilla)")
    eje.legend(); eje.grid(alpha=0.3)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
