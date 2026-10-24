"""
lectura.py
Lectura de particiones y de chips GeoTIFF de Sen1Floods11.
Convención de nombres: {Evento}_{idChip}_{Capa}.tif   ej. Bolivia_103757_S1Hand.tif
"""
import numpy as np
import pandas as pd
import rasterio

from .configuracion import DIR_PARTICIONES, DIR_S1_MANO, DIR_ETIQUETA_MANO, DIR_OTSU_MANO

PARTICIONES = ["train", "valid", "test", "bolivia"]


def leer_particion(nombre: str) -> pd.DataFrame:
    """Devuelve un DataFrame con id_chip, evento y partición a partir del CSV oficial."""
    ruta = DIR_PARTICIONES / f"flood_{nombre}_data.csv"
    df = pd.read_csv(ruta, header=None, names=["archivo_s1", "archivo_etiqueta"])
    df["id_chip"] = df["archivo_s1"].str.replace("_S1Hand.tif", "", regex=False)
    df["evento"] = df["id_chip"].str.split("_").str[0]
    df["particion"] = nombre
    return df[["id_chip", "evento", "particion"]]


def leer_todas_las_particiones() -> pd.DataFrame:
    return pd.concat([leer_particion(p) for p in PARTICIONES], ignore_index=True)


def leer_tif(ruta) -> np.ndarray:
    with rasterio.open(ruta) as src:
        return src.read()          # (bandas, alto, ancho)


def leer_s1(id_chip: str, carpeta=DIR_S1_MANO, sufijo: str = "S1Hand") -> np.ndarray:
    """SAR Sentinel-1: (2, 512, 512) float32 -> banda 0 = VV, banda 1 = VH, en dB."""
    return leer_tif(carpeta / f"{id_chip}_{sufijo}.tif").astype(np.float32)


def leer_etiqueta(id_chip: str, carpeta=DIR_ETIQUETA_MANO, sufijo: str = "LabelHand") -> np.ndarray:
    """Máscara (512, 512) int16: -1 sin dato, 0 no agua, 1 agua."""
    return leer_tif(carpeta / f"{id_chip}_{sufijo}.tif")[0].astype(np.int16)


def leer_otsu(id_chip: str) -> np.ndarray:
    """Etiqueta automática por umbral de Otsu sobre el MISMO chip (la 'etiqueta débil')."""
    return leer_etiqueta(id_chip, DIR_OTSU_MANO, "S1OtsuLabelHand")
