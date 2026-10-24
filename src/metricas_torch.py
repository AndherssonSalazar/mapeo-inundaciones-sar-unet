"""
metricas_torch.py
Acumulador de la matriz de confusión durante la evaluación (en GPU, sin copiar píxeles a CPU).
Clave: las métricas se calculan con los conteos SUMADOS de todos los chips (IoU agregado),
no promediando IoU chip por chip -> los chips sin agua (26 en Ghana) no producen 0/0.
También acumula por evento, para el desglose del informe.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .metricas import metricas_desde_conteos

VALOR_IGNORAR = -1


class AcumuladorConfusion:
    def __init__(self, umbral: float = 0.5):
        self.umbral = umbral
        self.reiniciar()

    def reiniciar(self):
        self.global_ = torch.zeros(4, dtype=torch.int64)           # vp, fp, fn, vn
        self.por_evento = defaultdict(lambda: torch.zeros(4, dtype=torch.int64))

    @torch.no_grad()
    def actualizar(self, logits, etiquetas, ids_chips=None):
        if logits.dim() == 4:
            logits = logits.squeeze(1)
        prediccion = torch.sigmoid(logits.float()) > self.umbral
        referencia = etiquetas == 1
        validos = etiquetas != VALOR_IGNORAR
        # conteos por chip (dim 1,2 = alto, ancho) -> (B, 4)
        conteos = torch.stack([
            (prediccion & referencia & validos).sum(dim=(1, 2)),
            (prediccion & ~referencia & validos).sum(dim=(1, 2)),
            (~prediccion & referencia & validos).sum(dim=(1, 2)),
            (~prediccion & ~referencia & validos).sum(dim=(1, 2)),
        ], dim=1).cpu()
        self.global_ += conteos.sum(dim=0)
        if ids_chips is not None:
            for id_chip, fila in zip(ids_chips, conteos):
                self.por_evento[id_chip.split("_")[0]] += fila

    def metricas(self) -> dict:
        return metricas_desde_conteos(*self.global_.tolist())

    def matriz_confusion(self) -> np.ndarray:
        """[[VN, FP], [FN, VP]]  (filas = real no agua / agua; columnas = predicho no agua / agua)."""
        vp, fp, fn, vn = self.global_.tolist()
        return np.array([[vn, fp], [fn, vp]], dtype=np.int64)

    def tabla_por_evento(self) -> pd.DataFrame:
        filas = {evento: metricas_desde_conteos(*c.tolist()) for evento, c in self.por_evento.items()}
        return pd.DataFrame(filas).T.sort_values("iou")
