"""Paso 2 — Estadísticas de normalización del SAR (p1, p99, media, desviación), SOLO con train."""
import json

import _entorno as E
from src.configuracion import DIR_RESULTADOS, SEMILLA
from src.exploracion import muestrear_pixeles, estadisticas_bandas
from src.lectura import leer_particion

E.titulo("Paso 2/9 — Estadísticas de normalización (solo train)")
ids_train = leer_particion("train")["id_chip"]
# muestreo NO balanceado: la red ve la proporción real de agua, así que se normaliza con ella
pixeles = muestrear_pixeles(ids_train, n_por_clase=E.PARAMETROS["pixeles_por_chip"], semilla=SEMILLA, balanceado=False)
estadisticas = estadisticas_bandas(pixeles)
with open(DIR_RESULTADOS / "f1_estadisticas_normalizacion.json", "w") as f:
    json.dump(estadisticas, f, indent=2)
print(json.dumps(estadisticas, indent=2))
