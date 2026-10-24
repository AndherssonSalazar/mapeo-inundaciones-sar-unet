"""
visualizacion_fallos.py
Una figura por tipo de error: fila de paneles heterogéneos (SAR, óptico, probabilidad, errores, capas físicas).
"""
import matplotlib.pyplot as plt
import numpy as np

from .visualizacion import MAPA_ETIQUETA, NORMA_ETIQUETA
from .visualizacion_final import MAPA_ERRORES, LEYENDA


def _dibujar(eje, tipo, datos, opciones):
    if tipo == "sar":
        eje.imshow(np.nan_to_num(datos, nan=opciones.get("vmin", -25)), cmap="gray",
                   vmin=opciones.get("vmin", -25), vmax=opciones.get("vmax", 0))
    elif tipo == "rgb":
        eje.imshow(datos)
    elif tipo == "prob":
        eje.imshow(datos, cmap="Blues", vmin=0, vmax=1)
    elif tipo == "errores":
        eje.imshow(datos, cmap=MAPA_ERRORES, vmin=0, vmax=4, interpolation="nearest")
    elif tipo == "etiqueta":
        eje.imshow(datos, cmap=MAPA_ETIQUETA, norm=NORMA_ETIQUETA, interpolation="nearest")
    elif tipo in ("divergente", "continuo"):
        mapa = "RdBu_r" if tipo == "divergente" else opciones.get("cmap", "viridis")
        imagen = eje.imshow(datos, cmap=mapa, vmin=opciones.get("vmin"), vmax=opciones.get("vmax"))
        plt.colorbar(imagen, ax=eje, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
    if "contorno" in opciones:                      # p. ej. borde de la etiqueta de agua
        eje.contour(opciones["contorno"], levels=[0.5], colors=opciones.get("color_contorno", "yellow"), linewidths=0.8)


def figura_caso(titulo, paneles, ruta_guardado=None, leyenda_errores=True):
    """paneles: [(tipo, datos, subtítulo, {opciones}), ...]"""
    figura, ejes = plt.subplots(1, len(paneles), figsize=(3.4 * len(paneles), 3.9))
    for eje, (tipo, datos, subtitulo, opciones) in zip(np.atleast_1d(ejes), paneles):
        _dibujar(eje, tipo, datos, opciones)
        eje.set_title(subtitulo, fontsize=9)
        eje.axis("off")
    figura.suptitle(titulo, fontsize=11, fontweight="bold")
    if leyenda_errores:
        figura.legend(handles=LEYENDA, loc="lower center", ncol=4, frameon=False, fontsize=8)
    figura.tight_layout(rect=(0, 0.06 if leyenda_errores else 0, 1, 0.95))
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
