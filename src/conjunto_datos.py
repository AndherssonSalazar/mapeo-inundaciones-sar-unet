"""
conjunto_datos.py
Dataset de PyTorch para Sen1Floods11.
Devuelve (imagen, etiqueta, id_chip):
  imagen   : tensor float32 (2, H, W)  -> VV, VH normalizados
  etiqueta : tensor int64   (H, W)     -> 0 no agua, 1 agua, -1 IGNORAR (sin dato o SAR no finito)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .aumentos import aumentar, recorte_aleatorio
from .configuracion import VALOR_SIN_DATO, DIR_S1_MANO, DIR_ETIQUETA_MANO
from .lectura import leer_s1, leer_etiqueta


class ConjuntoInundaciones(Dataset):
    def __init__(self, ids_chips, normalizador, aumentar_datos: bool = False,
                 tamano_recorte: int | None = None, en_memoria: bool = True,
                 dir_s1=DIR_S1_MANO, sufijo_s1: str = "S1Hand",
                 dir_etiqueta=DIR_ETIQUETA_MANO, sufijo_etiqueta: str = "LabelHand",
                 semilla: int = 42):
        self.ids_chips = list(ids_chips)
        self.normalizador = normalizador
        self.aumentar_datos = aumentar_datos
        self.tamano_recorte = tamano_recorte
        self.dir_s1, self.sufijo_s1 = dir_s1, sufijo_s1
        self.dir_etiqueta, self.sufijo_etiqueta = dir_etiqueta, sufijo_etiqueta
        self.generador = np.random.default_rng(semilla)

        # Los chips a mano caben en RAM (~2 MB c/u): se cargan UNA vez, antes de que los
        # workers del DataLoader se creen (así lo comparten en vez de leer disco cada época).
        self.cache = None
        if en_memoria:
            self.cache = [self._cargar(i) for i in tqdm(self.ids_chips, desc="Cargando chips en RAM")]

    def _cargar(self, id_chip):
        s1 = leer_s1(id_chip, self.dir_s1, self.sufijo_s1)
        etiqueta = leer_etiqueta(id_chip, self.dir_etiqueta, self.sufijo_etiqueta)
        imagen, finitos = self.normalizador(s1)
        etiqueta = etiqueta.copy()
        etiqueta[~finitos] = VALOR_SIN_DATO          # SAR inválido -> no se evalúa ni se aprende
        return imagen, etiqueta.astype(np.int64)

    def __len__(self):
        return len(self.ids_chips)

    def __getitem__(self, indice):
        imagen, etiqueta = self.cache[indice] if self.cache is not None else self._cargar(self.ids_chips[indice])
        if self.tamano_recorte:
            imagen, etiqueta = recorte_aleatorio(imagen, etiqueta, self.tamano_recorte, self.generador)
        if self.aumentar_datos:
            imagen, etiqueta = aumentar(imagen, etiqueta, self.generador)
        return torch.from_numpy(np.ascontiguousarray(imagen)), \
               torch.from_numpy(np.ascontiguousarray(etiqueta)), self.ids_chips[indice]


def semilla_por_worker(id_worker: int):
    """Cada worker del DataLoader recibe su propio generador (si no, todos aumentarían igual)."""
    informacion = torch.utils.data.get_worker_info()
    semilla = (torch.initial_seed() + id_worker) % 2**32
    informacion.dataset.generador = np.random.default_rng(semilla)
