"""Paso 3 — Línea base: umbral de Otsu evaluado con las mismas métricas y máscara que la U-Net."""
import pandas as pd
import torch
from tqdm.auto import tqdm

import _entorno as E
from src.configuracion import DIR_RESULTADOS, DIR_FIGURAS
from src.conjunto_datos import ConjuntoInundaciones
from src.lectura import leer_particion, leer_otsu
from src.metricas_torch import AcumuladorConfusion
from src.visualizacion_metricas import figura_matriz_confusion

E.titulo("Paso 3/9 — Línea base de Otsu")
normalizador = E.normalizador()


def evaluar_otsu(particion):
    conjunto = ConjuntoInundaciones(leer_particion(particion)["id_chip"], normalizador, en_memoria=False)
    acumulador = AcumuladorConfusion()
    for i in tqdm(range(len(conjunto)), desc=f"Otsu {particion}"):
        _, etiqueta, id_chip = conjunto[i]
        logits = torch.where(torch.from_numpy(leer_otsu(id_chip)) == 1, 10.0, -10.0)
        acumulador.actualizar(logits[None], etiqueta[None], [id_chip])
    return acumulador


acumuladores = {p: evaluar_otsu(p) for p in ["valid", "test", "bolivia"]}
linea_base = pd.DataFrame({p: a.metricas() for p, a in acumuladores.items()}).T
linea_base.to_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv")
figura_matriz_confusion(acumuladores["test"].matriz_confusion(), "Otsu — test",
                        DIR_FIGURAS / "f4_matriz_confusion_otsu_test.png")
print(linea_base.round(3))
