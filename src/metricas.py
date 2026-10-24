"""
metricas.py
Métricas de segmentación binaria para la clase "agua" a partir de conteos de la matriz de confusión.
Siempre se calculan SOLO sobre píxeles válidos (etiqueta != -1).
"""
import numpy as np


def conteos_confusion(prediccion: np.ndarray, referencia: np.ndarray, validos: np.ndarray) -> dict:
    """prediccion/referencia: arrays binarios {0,1}; validos: máscara booleana."""
    p = prediccion[validos].astype(bool)
    r = referencia[validos].astype(bool)
    return {
        "vp": int(np.sum(p & r)),     # verdaderos positivos (agua bien detectada)
        "fp": int(np.sum(p & ~r)),    # falsos positivos (dice agua y no es)
        "fn": int(np.sum(~p & r)),    # falsos negativos (agua no detectada)
        "vn": int(np.sum(~p & ~r)),   # verdaderos negativos
    }


def metricas_desde_conteos(vp: int, fp: int, fn: int, vn: int) -> dict:
    eps = 1e-9
    precision = vp / (vp + fp + eps)
    recall = vp / (vp + fn + eps)
    return {
        "iou": vp / (vp + fp + fn + eps),
        "f1": 2 * precision * recall / (precision + recall + eps),
        "precision": precision,
        "recall": recall,
        "omision": fn / (vp + fn + eps),     # = 1 - recall   (agua que se nos escapó)
        "comision": fp / (vp + fp + eps),    # = 1 - precision (agua inventada)
        "exactitud": (vp + vn) / (vp + fp + fn + vn + eps),
    }
