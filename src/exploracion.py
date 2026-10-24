"""
exploracion.py
Funciones del análisis exploratorio (fase 1): balance de clases, estadísticas SAR
y ruido de la etiqueta débil (Otsu) frente a la dorada (mano).
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .configuracion import VALOR_SIN_DATO
from .lectura import leer_s1, leer_etiqueta, leer_otsu
from .metricas import conteos_confusion


def resumen_chip(id_chip: str) -> dict:
    s1 = leer_s1(id_chip)
    etiqueta = leer_etiqueta(id_chip)
    otsu = leer_otsu(id_chip)

    s1_finito = np.isfinite(s1).all(axis=0)
    validos = (etiqueta != VALOR_SIN_DATO) & (otsu != VALOR_SIN_DATO) & s1_finito

    fila = {
        "id_chip": id_chip,
        "px_agua": int(np.sum(etiqueta == 1)),
        "px_no_agua": int(np.sum(etiqueta == 0)),
        "px_sin_dato": int(np.sum(etiqueta == VALOR_SIN_DATO)),
        "px_s1_no_finito": int(np.sum(~s1_finito)),
    }
    fila.update(conteos_confusion(otsu == 1, etiqueta == 1, validos))
    return fila


def tabla_resumen(df_chips: pd.DataFrame) -> pd.DataFrame:
    ids_unicos = df_chips["id_chip"].unique()
    filas = [resumen_chip(i) for i in tqdm(ids_unicos, desc="Resumiendo chips")]
    return df_chips.merge(pd.DataFrame(filas), on="id_chip", validate="many_to_one")


def muestrear_pixeles(ids_chips, n_por_clase: int = 2000, semilla: int = 42,
                      balanceado: bool = True) -> pd.DataFrame:
    """Devuelve un DataFrame (vv, vh, clase) con píxeles muestreados de cada chip.
    - balanceado=True : hasta n píxeles de agua y n de no agua por chip (para comparar las
                        DISTRIBUCIONES de cada clase en los histogramas).
    - balanceado=False: n píxeles válidos al azar por chip, respetando la proporción real
                        de clases (para estadísticas de normalización sin sesgo).
    Muestrear evita cargar 446 x 262k píxeles en memoria."""
    generador = np.random.default_rng(semilla)
    partes = []
    for id_chip in tqdm(ids_chips, desc="Muestreando píxeles"):
        s1 = leer_s1(id_chip)
        etiqueta = leer_etiqueta(id_chip)
        finito = np.isfinite(s1).all(axis=0)
        grupos = [(etiqueta == 0) & finito, (etiqueta == 1) & finito] if balanceado \
            else [(etiqueta != VALOR_SIN_DATO) & finito]
        for mascara in grupos:
            filas, columnas = np.where(mascara)
            if len(filas) == 0:
                continue
            elegidos = generador.choice(len(filas), size=min(n_por_clase, len(filas)), replace=False)
            f, c = filas[elegidos], columnas[elegidos]
            partes.append(pd.DataFrame({"vv": s1[0, f, c], "vh": s1[1, f, c], "clase": etiqueta[f, c]}))
    return pd.concat(partes, ignore_index=True)


def estadisticas_bandas(df_pixeles: pd.DataFrame) -> dict:
    """Percentiles, media y desviación por banda (para normalizar en la fase 2)."""
    resultado = {}
    for banda in ["vv", "vh"]:
        valores = df_pixeles[banda].to_numpy()
        resultado[banda] = {
            "p1": float(np.percentile(valores, 1)),
            "p99": float(np.percentile(valores, 99)),
            "media": float(valores.mean()),
            "desviacion": float(valores.std()),
        }
    return resultado
