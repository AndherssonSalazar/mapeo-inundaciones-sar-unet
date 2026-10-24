"""
visualizacion.py
Figuras reutilizables (exploración ahora; predicciones y fallos en fases posteriores).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# -1 sin dato (gris), 0 no agua (arena), 1 agua (azul)
MAPA_ETIQUETA = ListedColormap(["#9e9e9e", "#efe6d2", "#1f5fbf"])
NORMA_ETIQUETA = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], MAPA_ETIQUETA.N)
RANGO_DB = {"VV": (-25, 0), "VH": (-32, -5)}   # rango de visualización típico en dB


def mostrar_mascara(eje, mascara, titulo):
    eje.imshow(mascara, cmap=MAPA_ETIQUETA, norm=NORMA_ETIQUETA, interpolation="nearest")
    eje.set_title(titulo, fontsize=9)
    eje.axis("off")


def mostrar_sar(eje, banda, nombre):
    vmin, vmax = RANGO_DB[nombre]
    eje.imshow(np.nan_to_num(banda, nan=vmin), cmap="gray", vmin=vmin, vmax=vmax)
    eje.set_title(f"{nombre} (dB)", fontsize=9)
    eje.axis("off")


def figura_chips(lista_chips, ruta_guardado=None):
    """lista_chips: [(id_chip, s1, etiqueta, otsu), ...] -> una fila por chip."""
    figura, ejes = plt.subplots(len(lista_chips), 4, figsize=(12, 3.1 * len(lista_chips)))
    ejes = np.atleast_2d(ejes)
    for fila, (id_chip, s1, etiqueta, otsu) in enumerate(lista_chips):
        mostrar_sar(ejes[fila, 0], s1[0], "VV")
        mostrar_sar(ejes[fila, 1], s1[1], "VH")
        mostrar_mascara(ejes[fila, 2], etiqueta, "Etiqueta dorada (mano)")
        mostrar_mascara(ejes[fila, 3], otsu, "Etiqueta débil (Otsu)")
        ejes[fila, 0].text(-0.08, 0.5, id_chip, transform=ejes[fila, 0].transAxes,
                           rotation=90, va="center", ha="right", fontsize=9)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura


def figura_histogramas(df_pixeles, ruta_guardado=None):
    """Distribución de retrodispersión (dB) para agua vs no agua, por banda."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    for eje, banda in zip(ejes, ["vv", "vh"]):
        for clase, color, nombre in [(0, "#b08d57", "No agua"), (1, "#1f5fbf", "Agua")]:
            valores = df_pixeles.loc[df_pixeles["clase"] == clase, banda]
            eje.hist(valores, bins=120, range=(-40, 5), density=True, alpha=0.6,
                     color=color, label=nombre)
        eje.set_xlabel(f"{banda.upper()} (dB)")
        eje.set_ylabel("Densidad")
        eje.legend()
        eje.grid(alpha=0.3)
    figura.tight_layout()
    if ruta_guardado:
        figura.savefig(ruta_guardado, dpi=150, bbox_inches="tight")
    return figura
