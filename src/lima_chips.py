"""
lima_chips.py
Fase 8b: construir la región de prueba de Lima con el MISMO formato que Sen1Floods11.
  - Grilla EPSG:4326 con la resolución de Sen1Floods11 (8.983e-05° ≈ 10 m), chips de 512 x 512.
  - S1 (VV, VH en dB) descargado chip por chip desde Earth Engine con ee.data.computePixels
    (sin exportar a Drive; cada chip ~2 MB, muy por debajo del límite de 48 MB por pedido).
  - Etiqueta rasterizada desde EMSN038:
        -1 : fuera del área observada por WorldView-2 o cauce dudoso
         0 : observado y no inundado
         1 : extensión máxima de inundación (Potential_Maximum_flood)
  Se guardan como Lima_<fila>_<col>_S1Hand.tif / _LabelHand.tif para reutilizar ConjuntoInundaciones.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
import shapely
from shapely.ops import unary_union

RESOLUCION_SEN1FLOODS11 = 8.983152841195215e-05   # grados (verificado en la fase 0)
TAMANO = 512
SIN_DATO_EE = -9999.0


# ----------------------------------------------------------------------------- geometrías EMS
def preparar_geometrias(dir_ems, gdb="EMSN038_UTM18S_v0.gdb", cauce="ignorar"):
    """Devuelve las geometrías (en EPSG:4326) que definen la etiqueta.
    cauce: qué hacer con WatrcrsA fuera de la inundación -> 'ignorar' (-1), 'no_agua' (0) o 'agua' (1)."""
    import geopandas as gpd
    leer = lambda capa: gpd.read_file(dir_ems / gdb, layer=capa)
    aoi = leer("EMSN038_AOI")
    huellas = leer("Footprints")
    huellas_wv2 = huellas[huellas["SDP"].astype(str).str.contains("WorldView", case=False)]
    observado = shapely.force_2d(unary_union(huellas_wv2.geometry)).intersection(
        shapely.force_2d(unary_union(aoi.geometry)))       # el AOI viene en 3D (Z/M)
    inundacion = unary_union(leer("Potential_Maximum_flood").geometry)
    cauces = unary_union(leer("WatrcrsA").geometry)
    serie = gpd.GeoSeries([observado, inundacion, cauces], crs=aoi.crs).to_crs(4326)
    return {"observado": serie.iloc[0], "inundacion": serie.iloc[1], "cauces": serie.iloc[2],
            "cauce": cauce, "fechas_wv2": sorted(huellas_wv2["SDV"].astype(str).unique())}


def rasterizar_etiqueta(geometrias, transformacion, tamano=TAMANO):
    forma = (tamano, tamano)
    etiqueta = np.full(forma, -1, dtype=np.int16)
    quemar = lambda geom, valor: rasterize([(geom, valor)], out_shape=forma, transform=transformacion,
                                           fill=0, dtype="int16", all_touched=False).astype(bool)
    observado = quemar(geometrias["observado"], 1)
    etiqueta[observado] = 0
    if not geometrias["cauces"].is_empty:
        valor_cauce = {"ignorar": -1, "no_agua": 0, "agua": 1}[geometrias["cauce"]]
        etiqueta[quemar(geometrias["cauces"], 1) & observado] = valor_cauce
    etiqueta[quemar(geometrias["inundacion"], 1) & observado] = 1
    return etiqueta


# ----------------------------------------------------------------------------- grilla y S1
def definir_grilla(limites, resolucion=RESOLUCION_SEN1FLOODS11, tamano=TAMANO):
    """limites = (lon_min, lat_min, lon_max, lat_max) -> lista de (fila, col, Affine) cubriendo el área."""
    lon_min, lat_min, lon_max, lat_max = limites
    paso = resolucion * tamano
    n_col = int(np.ceil((lon_max - lon_min) / paso))
    n_fil = int(np.ceil((lat_max - lat_min) / paso))
    return [(f, c, Affine(resolucion, 0, lon_min + c * paso, 0, -resolucion, lat_max - f * paso))
            for f in range(n_fil) for c in range(n_col)]


def descargar_chip_s1(ee, imagen, transformacion, tamano=TAMANO):
    """VV y VH de un chip como (2, H, W) float32 en dB; NaN donde no hay dato."""
    pedido = {
        "expression": imagen.select(["VV", "VH"]).unmask(SIN_DATO_EE),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": tamano, "height": tamano},
                 "affineTransform": {"scaleX": transformacion.a, "shearX": 0, "translateX": transformacion.c,
                                     "shearY": 0, "scaleY": transformacion.e, "translateY": transformacion.f},
                 "crsCode": "EPSG:4326"},
    }
    datos = ee.data.computePixels(pedido)
    s1 = np.stack([datos["VV"], datos["VH"]]).astype(np.float32)
    s1[s1 <= SIN_DATO_EE + 1] = np.nan
    return s1


def escribir_tif(ruta, arreglo, transformacion, dtype):
    arreglo = arreglo if arreglo.ndim == 3 else arreglo[None]
    with rasterio.open(ruta, "w", driver="GTiff", height=arreglo.shape[1], width=arreglo.shape[2],
                       count=arreglo.shape[0], dtype=dtype, crs="EPSG:4326", transform=transformacion,
                       compress="deflate") as destino:
        destino.write(arreglo.astype(dtype))


def construir_chips(ee, imagen, geometrias, dir_s1, dir_etiqueta, minimo_observado=0.10):
    """Recorre la grilla sobre el área observada; guarda solo chips con >= minimo_observado de píxeles válidos."""
    from tqdm.auto import tqdm
    dir_s1.mkdir(parents=True, exist_ok=True); dir_etiqueta.mkdir(parents=True, exist_ok=True)
    filas = []
    for fila, col, transformacion in tqdm(definir_grilla(geometrias["observado"].bounds), desc="Chips Lima"):
        id_chip = f"Lima_{fila:02d}{col:02d}"
        etiqueta = rasterizar_etiqueta(geometrias, transformacion)
        fraccion_observada = (etiqueta != -1).mean()
        if fraccion_observada < minimo_observado:
            continue
        ruta_s1 = dir_s1 / f"{id_chip}_S1Hand.tif"
        if ruta_s1.exists():
            with rasterio.open(ruta_s1) as fuente:
                s1 = fuente.read()
        else:
            s1 = descargar_chip_s1(ee, imagen, transformacion)
            escribir_tif(ruta_s1, s1, transformacion, "float32")
        etiqueta[~np.isfinite(s1).all(axis=0)] = -1
        escribir_tif(dir_etiqueta / f"{id_chip}_LabelHand.tif", etiqueta, transformacion, "int16")
        validos = etiqueta != -1
        filas.append({"id_chip": id_chip, "fila": fila, "col": col,
                      "fraccion_valida": validos.mean(),
                      "px_agua": int((etiqueta == 1).sum()), "px_no_agua": int((etiqueta == 0).sum()),
                      "pct_agua_validos": 100 * (etiqueta == 1).sum() / max(validos.sum(), 1)})
    return pd.DataFrame(filas)
