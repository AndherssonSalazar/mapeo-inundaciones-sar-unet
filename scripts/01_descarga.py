"""Paso 1 — Descarga de Sen1Floods11: chips a mano (S1, etiquetas, Otsu, S2) y chips débiles."""
import os

import _entorno as E
from _descarga_gcs import descargar_mano, descargar_debil

E.titulo("Paso 1/9 — Descarga de Sen1Floods11")
descargar_mano(con_s2=True)
if os.environ.get("RNAP_SIN_DEBILES") == "1":
    print("Chips débiles omitidos (--sin-debiles): el paso 5 se salta.")
else:
    descargar_debil(limite=E.PARAMETROS["debiles_a_descargar"])
print("Descarga lista.")
