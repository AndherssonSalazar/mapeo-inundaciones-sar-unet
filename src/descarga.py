"""
descarga.py
Descarga de Sen1Floods11 desde el bucket público al disco local de Colab.
Se llama al inicio de cada notebook: si los datos ya están, no hace nada.
"""
import subprocess

from .configuracion import (BUCKET, DIR_S1_MANO, DIR_ETIQUETA_MANO, DIR_OTSU_MANO,
                            DIR_S2_MANO, DIR_S1_DEBIL, DIR_ETIQUETA_DEBIL, DIR_PARTICIONES)

_ORIGENES = {
    DIR_S1_MANO:        "data/flood_events/HandLabeled/S1Hand",
    DIR_ETIQUETA_MANO:  "data/flood_events/HandLabeled/LabelHand",
    DIR_OTSU_MANO:      "data/flood_events/HandLabeled/S1OtsuLabelHand",
    DIR_PARTICIONES:    "splits/flood_handlabeled",
    DIR_S2_MANO:        "data/flood_events/HandLabeled/S2Hand",
    DIR_S1_DEBIL:       "data/flood_events/WeaklyLabeled/S1Weak",
    DIR_ETIQUETA_DEBIL: "data/flood_events/WeaklyLabeled/S1OtsuLabelWeak",
}


def _descargar(destino) -> None:
    if destino.exists() and any(destino.iterdir()):
        print(f"[ya existe] {destino.name}")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"[descargando] {destino.name} ...")
    subprocess.run(["gsutil", "-m", "-q", "cp", "-r", f"{BUCKET}/{_ORIGENES[destino]}",
                    str(destino.parent)], check=True)


def descargar_mano() -> None:
    """Chips etiquetados a mano (S1, etiqueta dorada, etiqueta Otsu) + particiones."""
    for destino in [DIR_S1_MANO, DIR_ETIQUETA_MANO, DIR_OTSU_MANO, DIR_PARTICIONES]:
        _descargar(destino)


def descargar_debil() -> None:
    """Chips con etiqueta débil (~9 GB). Recién en la fase 6."""
    for destino in [DIR_S1_DEBIL, DIR_ETIQUETA_DEBIL]:
        _descargar(destino)


def descargar_s2_mano() -> None:
    """Óptico Sentinel-2 de los chips a mano. Solo para el bono (fase 10)."""
    _descargar(DIR_S2_MANO)
