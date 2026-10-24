"""
visualizacion_entrenamiento.py
Curvas de entrenamiento y predicciones de ejemplo.
"""
import matplotlib.pyplot as plt
import numpy as np

from .visualizacion import mostrar_mascara


def figura_curvas(historial, titulo="", linea_base_iou=None, ruta_guardado=None):
    figura, ejes = plt.subplots(1, 3, figsize=(15, 3.8))
    ejes[0].plot(historial["epoca"], historial["perdida_train"], label="train")
    ejes[0].plot(historial["epoca"], historial["perdida_valid"], label="valid")
    ejes[0].set_title("Pérdida"); ejes[0].legend()
    ejes[1].plot(historial["epoca"], historial["iou_valid"], label="IoU", color="#1f5fbf")
    ejes[1].plot(historial["epoca"], historial["f1_valid"], label="F1", color="#7a9cc6")
    if linea_base_iou is not None:
        ejes[1].axhline(linea_base_iou, ls="--", color="gray", label=f"Otsu IoU = {linea_base_iou:.3f}")
    mejor = historial["iou_valid"].idxmax()
    ejes[1].scatter(historial["epoca"][mejor], historial["iou_valid"][mejor], color="red", zorder=5,
                    label=f"mejor: {historial['iou_valid'][mejor]:.3f} (ép. {historial['epoca'][mejor]})")
    ejes[1].set_title("Validación"); ejes[1].legend(fontsize=8)
    ejes[2].plot(historial["epoca"], historial["tasa"], color="black")
    ejes[2].set_title("Tasa de aprendizaje")
    for eje in ejes:
        eje.set_xlabel("época"); eje.grid(alpha=0.3)
    figura.suptitle(titulo)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura


def figura_predicciones(ejemplos, umbral=0.5, ruta_guardado=None):
    """ejemplos: [(id_chip, vh_normalizado, etiqueta, probabilidad), ...]"""
    figura, ejes = plt.subplots(len(ejemplos), 4, figsize=(12, 3.1 * len(ejemplos)))
    ejes = np.atleast_2d(ejes)
    for fila, (id_chip, vh, etiqueta, probabilidad) in enumerate(ejemplos):
        ejes[fila, 0].imshow(vh, cmap="gray", vmin=-2.5, vmax=2.5); ejes[fila, 0].set_title("VH normalizado", fontsize=9)
        mostrar_mascara(ejes[fila, 1], etiqueta, "Etiqueta dorada")
        ejes[fila, 2].imshow(probabilidad, cmap="Blues", vmin=0, vmax=1); ejes[fila, 2].set_title("P(agua)", fontsize=9)
        prediccion = np.where(etiqueta == -1, -1, (probabilidad > umbral).astype(int))
        mostrar_mascara(ejes[fila, 3], prediccion, f"Predicción (>{umbral})")
        for eje in ejes[fila, :3]:
            eje.axis("off")
        ejes[fila, 0].text(-0.08, 0.5, id_chip, transform=ejes[fila, 0].transAxes, rotation=90,
                           va="center", ha="right", fontsize=9)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
