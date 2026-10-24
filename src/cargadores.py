"""
cargadores.py
Crea los DataLoader de las 4 particiones oficiales de los chips a mano.
Solo 'train' lleva aumento de datos (y recorte opcional); valid/test/bolivia se evalúan tal cual.
"""
import torch
from torch.utils.data import DataLoader

from .conjunto_datos import ConjuntoInundaciones, semilla_por_worker
from .lectura import leer_particion


def crear_cargadores(normalizador, tamano_lote: int = 8, tamano_recorte: int | None = None,
                     num_workers: int = 2, semilla: int = 42) -> dict:
    cargadores = {}
    for particion in ["train", "valid", "test", "bolivia"]:
        es_train = particion == "train"
        ids = leer_particion(particion)["id_chip"]
        conjunto = ConjuntoInundaciones(ids, normalizador, aumentar_datos=es_train,
                                        tamano_recorte=tamano_recorte if es_train else None,
                                        semilla=semilla)
        cargadores[particion] = DataLoader(
            conjunto,
            batch_size=tamano_lote if es_train else max(1, tamano_lote // 2),
            shuffle=es_train,
            drop_last=es_train,              # evita un último lote de 1-2 chips (BatchNorm inestable)
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=semilla_por_worker,
            generator=torch.Generator().manual_seed(semilla),
            persistent_workers=num_workers > 0,
        )
        print(f"{particion:8s}: {len(conjunto):3d} chips -> {len(cargadores[particion]):3d} lotes")
    return cargadores
