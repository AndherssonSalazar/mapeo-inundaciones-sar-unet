"""
_descarga_gcs.py
Descarga de Sen1Floods11 desde el bucket público de Google Cloud con el cliente ANÓNIMO de Python
(no necesita gsutil ni credenciales; funciona en Kaggle, Colab o una PC). Es la misma lógica que la
versión de Kaggle de src/descarga.py, con la opción de bajar solo una muestra de los chips débiles.
"""
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

from src.configuracion import (DIR_S1_MANO, DIR_ETIQUETA_MANO, DIR_OTSU_MANO, DIR_S2_MANO,
                               DIR_S1_DEBIL, DIR_ETIQUETA_DEBIL, DIR_PARTICIONES)

NOMBRE_BUCKET, VERSION = "sen1floods11", "v1.1"
ORIGENES = {
    DIR_S1_MANO:        "data/flood_events/HandLabeled/S1Hand",
    DIR_ETIQUETA_MANO:  "data/flood_events/HandLabeled/LabelHand",
    DIR_OTSU_MANO:      "data/flood_events/HandLabeled/S1OtsuLabelHand",
    DIR_PARTICIONES:    "splits/flood_handlabeled",
    DIR_S2_MANO:        "data/flood_events/HandLabeled/S2Hand",
    DIR_S1_DEBIL:       "data/flood_events/WeaklyLabeled/S1Weak",
    DIR_ETIQUETA_DEBIL: "data/flood_events/WeaklyLabeled/S1OtsuLabelWeak",
}


def _cliente():
    from google.cloud import storage
    return storage.Client.create_anonymous_client()


def _blobs(cliente, destino):
    prefijo = f"{VERSION}/{ORIGENES[destino]}/"
    return [b for b in cliente.list_blobs(NOMBRE_BUCKET, prefix=prefijo) if not b.name.endswith("/")]


def _bajar(blobs, destino, hilos: int = 16):
    destino.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(hilos) as ejecutor:
        list(tqdm(ejecutor.map(lambda b: b.download_to_filename(str(destino / b.name.split("/")[-1])), blobs),
                  total=len(blobs), desc=destino.name))


def descargar(destino) -> None:
    if destino.exists() and any(destino.iterdir()):
        print(f"[ya existe] {destino}")
        return
    _bajar(_blobs(_cliente(), destino), destino)


def descargar_mano(con_s2: bool = True) -> None:
    for destino in [DIR_S1_MANO, DIR_ETIQUETA_MANO, DIR_OTSU_MANO, DIR_PARTICIONES] + ([DIR_S2_MANO] if con_s2 else []):
        descargar(destino)


def descargar_debil(limite: int | None = None) -> None:
    """limite=None baja todos los chips débiles (~9 GB). Con limite=n baja solo n pares imagen-etiqueta
    (prueba rápida); excluye Bolivia, que nunca se usa para entrenar."""
    if limite is None:
        for destino in [DIR_S1_DEBIL, DIR_ETIQUETA_DEBIL]:
            descargar(destino)
        return
    if DIR_S1_DEBIL.exists() and any(DIR_S1_DEBIL.iterdir()):
        print(f"[ya existe] {DIR_S1_DEBIL}")
        return
    cliente = _cliente()
    imagenes = {b.name.split("/")[-1].replace("_S1Weak.tif", ""): b for b in _blobs(cliente, DIR_S1_DEBIL)}
    etiquetas = {b.name.split("/")[-1].replace("_S1OtsuLabelWeak.tif", ""): b for b in _blobs(cliente, DIR_ETIQUETA_DEBIL)}
    ids = sorted(i for i in imagenes.keys() & etiquetas.keys() if not i.startswith("Bolivia"))[::7][:limite]
    _bajar([imagenes[i] for i in ids], DIR_S1_DEBIL)
    _bajar([etiquetas[i] for i in ids], DIR_ETIQUETA_DEBIL)
