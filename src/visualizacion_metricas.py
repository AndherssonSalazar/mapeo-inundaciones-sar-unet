"""
visualizacion_metricas.py
Matriz de confusión comentable para el informe (entregable 3).
"""
import matplotlib.pyplot as plt
import numpy as np


def figura_matriz_confusion(matriz: np.ndarray, titulo: str = "", ruta_guardado=None):
    """matriz = [[VN, FP], [FN, VP]]. Muestra % por fila (cuánto de cada clase real se acierta) y conteos."""
    porcentajes = matriz / matriz.sum(axis=1, keepdims=True).clip(min=1)
    figura, eje = plt.subplots(figsize=(4.6, 4))
    eje.imshow(porcentajes, cmap="Blues", vmin=0, vmax=1)
    nombres = ["No agua", "Agua"]
    eje.set_xticks([0, 1], [f"Pred. {n}" for n in nombres])
    eje.set_yticks([0, 1], [f"Real {n}" for n in nombres])
    etiquetas = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if porcentajes[i, j] > 0.5 else "black"
            eje.text(j, i, f"{etiquetas[i][j]}\n{100 * porcentajes[i, j]:.1f}%\n({matriz[i, j]:,} px)",
                     ha="center", va="center", color=color, fontsize=9)
    eje.set_title(titulo)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
