"""
preprocesamiento.py
Normalización de las bandas SAR (VV, VH) con las estadísticas calculadas SOLO en train (fase 1).
Pasos:  1) detectar píxeles no finitos (NaN/inf)
        2) recortar a [p1, p99]  -> evita que valores extremos dominen
        3) estandarizar (x - media) / desviacion  -> entradas centradas en 0 y escala ~1
        4) NaN -> 0 (que tras estandarizar = "valor medio": la red no recibe basura)
"""
import json

import numpy as np

BANDAS = ["vv", "vh"]


class Normalizador:
    def __init__(self, estadisticas: dict):
        self.minimos = np.array([estadisticas[b]["p1"] for b in BANDAS], dtype=np.float32)[:, None, None]
        self.maximos = np.array([estadisticas[b]["p99"] for b in BANDAS], dtype=np.float32)[:, None, None]
        self.medias = np.array([estadisticas[b]["media"] for b in BANDAS], dtype=np.float32)[:, None, None]
        self.desviaciones = np.array([estadisticas[b]["desviacion"] for b in BANDAS], dtype=np.float32)[:, None, None]

    @classmethod
    def desde_json(cls, ruta) -> "Normalizador":
        with open(ruta) as f:
            return cls(json.load(f))

    def __call__(self, s1: np.ndarray):
        """s1: (2, H, W) en dB -> (s1 normalizado (2, H, W), máscara de píxeles finitos (H, W))."""
        finitos = np.isfinite(s1).all(axis=0)
        x = np.clip(s1, self.minimos, self.maximos)
        x = (x - self.medias) / self.desviaciones
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32), finitos

    def revertir(self, x: np.ndarray) -> np.ndarray:
        """De vuelta a dB (útil para visualizar lo que entra a la red)."""
        return x * self.desviaciones + self.medias
