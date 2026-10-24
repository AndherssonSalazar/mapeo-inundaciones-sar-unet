"""
visualizacion_lotes.py
Figuras para inspeccionar lo que REALMENTE entra a la red (después de normalizar y aumentar).
"""
import matplotlib.pyplot as plt
import numpy as np

from .visualizacion import mostrar_mascara


def figura_aumentos(conjunto, indice: int, repeticiones: int = 4, ruta_guardado=None):
    """Muestra el mismo chip con varias transformaciones aleatorias: VH normalizado arriba,
    máscara abajo. Si imagen y máscara están alineadas, el agua (oscura) cae sobre el azul."""
    figura, ejes = plt.subplots(2, repeticiones, figsize=(3 * repeticiones, 6))
    for j in range(repeticiones):
        imagen, etiqueta, id_chip = conjunto[indice]
        ejes[0, j].imshow(imagen[1].numpy(), cmap="gray", vmin=-2.5, vmax=2.5)
        ejes[0, j].set_title(f"VH normalizado #{j + 1}", fontsize=9)
        ejes[0, j].axis("off")
        mostrar_mascara(ejes[1, j], etiqueta.numpy(), "Máscara")
    figura.suptitle(id_chip)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura


def figura_histograma_normalizado(imagenes, etiquetas, ruta_guardado=None):
    """Histograma de las entradas normalizadas de un lote (debería verse centrado en ~0)."""
    figura, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
    validos = (etiquetas != -1).numpy()
    for b, (eje, nombre) in enumerate(zip(ejes, ["VV", "VH"])):
        valores = imagenes[:, b].numpy()
        for clase, color, texto in [(0, "#b08d57", "No agua"), (1, "#1f5fbf", "Agua")]:
            seleccion = valores[validos & (etiquetas.numpy() == clase)]
            if seleccion.size:
                eje.hist(seleccion, bins=80, range=(-4, 4), density=True, alpha=0.6, color=color, label=texto)
        eje.set_title(f"{nombre} normalizado")
        eje.legend()
        eje.grid(alpha=0.3)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
