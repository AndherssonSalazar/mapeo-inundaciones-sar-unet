"""
evaluacion_lima.py
Fase 8c (entregable 4): generalización geográfica.
  - ConjuntoLima: lee los chips de Lima con el MISMO preprocesamiento que Sen1Floods11, y además
    marca como -1 los píxeles con VV = VH = 0 dB exactos (sin dato que Earth Engine devolvió como 0
    en el borde de la franja del satélite). No modifica los archivos en disco.
  - Otsu por chip sobre VH como referencia clásica en Lima.
  - Evaluación de las 3 semillas del modelo final en cualquier región (test, Bolivia, Lima).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .conjunto_datos import ConjuntoInundaciones
from .configuracion import RAIZ_PROYECTO, VALOR_SIN_DATO
from .entrenamiento import cargar_mejor_modelo
from .evaluacion_final import evaluar_con_histograma, curva_umbral
from .lectura import leer_s1, leer_etiqueta
from .metricas_torch import AcumuladorConfusion

DIR_CHIPS_LIMA = RAIZ_PROYECTO / "datos_lima" / "chips"
DIR_S1_LIMA, DIR_ETIQUETA_LIMA = DIR_CHIPS_LIMA / "S1Hand", DIR_CHIPS_LIMA / "LabelHand"


def ids_lima():
    return sorted(r.name.replace("_S1Hand.tif", "") for r in DIR_S1_LIMA.glob("*_S1Hand.tif"))


def mascara_ceros(s1):
    """True donde VV y VH valen exactamente 0 dB (sin dato)."""
    return (s1 == 0).all(axis=0)


class ConjuntoLima(ConjuntoInundaciones):
    def __init__(self, ids_chips, normalizador, **parametros):
        super().__init__(ids_chips, normalizador, dir_s1=DIR_S1_LIMA, sufijo_s1="S1Hand",
                         dir_etiqueta=DIR_ETIQUETA_LIMA, sufijo_etiqueta="LabelHand", **parametros)

    def _cargar(self, id_chip):
        imagen, etiqueta = super()._cargar(id_chip)
        s1 = leer_s1(id_chip, self.dir_s1, self.sufijo_s1)
        etiqueta[mascara_ceros(s1)] = VALOR_SIN_DATO
        return imagen, etiqueta


def cargador_lima(normalizador, tamano_lote=4, num_workers=2):
    conjunto = ConjuntoLima(ids_lima(), normalizador)
    return DataLoader(conjunto, batch_size=tamano_lote, shuffle=False, num_workers=num_workers)


def diagnostico_ceros() -> pd.DataFrame:
    filas = []
    for id_chip in ids_lima():
        s1 = leer_s1(id_chip, DIR_S1_LIMA, "S1Hand")
        etiqueta = leer_etiqueta(id_chip, DIR_ETIQUETA_LIMA, "LabelHand")
        ceros = mascara_ceros(s1)
        filas.append({"id_chip": id_chip, "px_cero": int(ceros.sum()), "pct_chip": 100 * ceros.mean(),
                      "px_cero_etiquetados_agua": int((ceros & (etiqueta == 1)).sum()),
                      "px_cero_etiquetados_no_agua": int((ceros & (etiqueta == 0)).sum())})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------- referencia clásica
def otsu_lima(conjunto) -> AcumuladorConfusion:
    """Umbral de Otsu sobre VH (dB) en cada chip; agua = VH < umbral. Mismos píxeles ignorados que la U-Net."""
    from skimage.filters import threshold_otsu
    acumulador = AcumuladorConfusion()
    for i, id_chip in enumerate(conjunto.ids_chips):
        _, etiqueta, _ = conjunto[i]
        vh = leer_s1(id_chip, conjunto.dir_s1, conjunto.sufijo_s1)[1]
        validos = np.isfinite(vh) & (etiqueta.numpy() != VALOR_SIN_DATO)
        if validos.sum() < 100:
            continue
        umbral = threshold_otsu(vh[validos])
        logits = torch.from_numpy(np.where(np.nan_to_num(vh, nan=0) < umbral, 10.0, -10.0)).float()
        acumulador.actualizar(logits[None], etiqueta[None], [id_chip])
    return acumulador


# ----------------------------------------------------------------------------- U-Net en varias regiones
def evaluar_regiones(carpeta, semillas, cargadores: dict, dir_experimentos, dispositivo) -> dict:
    """cargadores = {'bolivia': DataLoader, 'lima': DataLoader, ...}. Umbral 0.5 + curvas IoU-umbral."""
    salida = {"filas": [], "acumuladores": {r: {} for r in cargadores}, "curvas": {r: {} for r in cargadores}}
    for semilla in semillas:
        modelo, _, _ = cargar_mejor_modelo(dir_experimentos / f"{carpeta}_s{semilla}", dispositivo)
        for region, cargador in cargadores.items():
            acumulador, hist_agua, hist_no_agua = evaluar_con_histograma(modelo, cargador, dispositivo)
            salida["filas"].append({"region": region, "semilla": semilla, **acumulador.metricas()})
            salida["acumuladores"][region][semilla] = acumulador
            salida["curvas"][region][semilla] = curva_umbral(hist_agua, hist_no_agua)
        del modelo
    salida["df"] = pd.DataFrame(salida["filas"])
    return salida
