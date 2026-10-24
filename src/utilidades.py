"""
utilidades.py
Funciones transversales: reproducibilidad y dispositivo.
"""
import os
import random

import numpy as np
import torch


def fijar_semilla(semilla: int = 42) -> None:
    """Fija todas las fuentes de aleatoriedad para que los experimentos sean repetibles
    (clave en el estudio de ablación: si cambia el resultado, que sea por la decisión, no por azar)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obtener_dispositivo() -> torch.device:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("OJO: sin GPU. En Colab: Entorno de ejecución > Cambiar tipo > T4 GPU")
    return dispositivo
