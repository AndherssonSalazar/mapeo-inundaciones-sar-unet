"""Paso 5 — E2(a): solo dorados vs. etiquetas débiles (solo débiles, mezcla, preentreno + ajuste),
todo con la pérdida combinada. La referencia 'solo dorados' reutiliza las corridas del paso 4."""
import os

import pandas as pd

import _entorno as E
from src.ablacion import correr_ablacion, resumir
from src.configuracion import DIR_RESULTADOS, DIR_FIGURAS
from src.utilidades import obtener_dispositivo
from src.visualizacion_ablacion import figura_ablacion

E.titulo("Paso 5/9 — Ablación (a): etiquetas débiles")
dispositivo = obtener_dispositivo()
normalizador = E.normalizador()
p, semillas = E.PARAMETROS, E.PARAMETROS["semillas"]
comunes = dict(tamano_lote=p["tamano_lote"], tamano_recorte=p["tamano_recorte"], num_workers=p["num_workers"])
base = E.configuracion_base(perdida="combinada")

df = correr_ablacion({"combinada": base}, semillas, normalizador, dispositivo, E.DIR_EXPERIMENTOS, **comunes)
df["variante"] = "solo_dorados"
if os.environ.get("RNAP_SIN_DEBILES") == "1":
    print("--sin-debiles: solo se registra la referencia 'solo dorados'.")
else:
    from src.ablacion_debil import listar_chips_debiles, correr_ablacion_debil
    chips = listar_chips_debiles()
    ids_debiles = chips.loc[chips["excluido"] == "", "id_chip"].tolist()
    print(f"Chips débiles usables: {len(ids_debiles)} (se excluyen Bolivia y los que coinciden con chips a mano)")
    df_debil = correr_ablacion_debil(base, semillas, normalizador, ids_debiles, dispositivo, E.DIR_EXPERIMENTOS,
                                     debiles_mezcla=p["debiles_mezcla"], debiles_preentreno=p["debiles_preentreno"],
                                     epocas_preentreno=p["epocas_preentreno"], **comunes)
    df = pd.concat([df, df_debil], ignore_index=True)

df.to_csv(DIR_RESULTADOS / "f6a_ablacion_corridas.csv", index=False)
resumen = resumir(df)
resumen.to_csv(DIR_RESULTADOS / "f6a_ablacion_resumen.csv")
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0).loc["valid", "iou"]
figura_ablacion(df, otsu, "Ablación (a) — dorados vs. débiles (validación)", DIR_FIGURAS / "f6a_ablacion.png")
print(resumen)
