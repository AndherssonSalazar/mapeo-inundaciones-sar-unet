"""
analisis_fallos.py
Fase 9 (entregable 5): evidencias físicas para explicar los errores característicos del SAR.

Todas las capas auxiliares se descargan con Earth Engine en la MISMA grilla del chip (misma transformación
afín que el GeoTIFF de Sentinel-1), así se superponen píxel a píxel con la predicción:
  - SRTM (pendiente y orientación) + ángulo de incidencia de S1 -> ángulo de incidencia LOCAL
  - Sentinel-2 (compuesto sin nubes) -> contexto óptico en color real
  - JRC Global Surface Water (ocurrencia) -> agua permanente
  - Sentinel-1 de otra fecha -> diferencia temporal (detección de cambio)
"""
import subprocess

import numpy as np
import rasterio

from .configuracion import BUCKET, DIR_S2_MANO

SIN_DATO = -9999.0
# Pasada descendente de Sentinel-1: el satélite viaja hacia el SSO y mira a su derecha (hacia el ONO).
# Azimut aproximado desde el terreno HACIA el sensor (~ESE). Supuesto documentado: ±5°.
AZIMUT_HACIA_SENSOR_DESCENDENTE = 103.0


# ----------------------------------------------------------------------------- utilidades de grilla
def transformacion_de(ruta_tif):
    with rasterio.open(ruta_tif) as fuente:
        return fuente.transform


def descargar_ee(ee, imagen, bandas, transformacion, tamano=512) -> dict:
    """Bandas de una ee.Image en la grilla exacta del chip -> {banda: array float32 con NaN donde no hay dato}."""
    pedido = {
        "expression": imagen.select(bandas).toFloat().unmask(SIN_DATO),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": tamano, "height": tamano},
                 "affineTransform": {"scaleX": transformacion.a, "shearX": 0, "translateX": transformacion.c,
                                     "shearY": 0, "scaleY": transformacion.e, "translateY": transformacion.f},
                 "crsCode": "EPSG:4326"},
    }
    datos = ee.data.computePixels(pedido)
    salida = {}
    for banda in bandas:
        arreglo = np.asarray(datos[banda], dtype=np.float32)
        arreglo[arreglo <= SIN_DATO + 1] = np.nan
        salida[banda] = arreglo
    return salida


def _region(ee, transformacion, tamano=512):
    x0, y1 = transformacion.c, transformacion.f
    x1, y0 = x0 + transformacion.a * tamano, y1 + transformacion.e * tamano
    return ee.Geometry.Rectangle([x0, y0, x1, y1])


# ----------------------------------------------------------------------------- relieve y geometría radar
def angulo_incidencia_local(pendiente, orientacion, incidencia, azimut_sensor=AZIMUT_HACIA_SENSOR_DESCENDENTE):
    """θ_local = acos( cos s·cos θ + sin s·sin θ·cos(a − φ_sensor) )  [grados]
    s = pendiente, a = orientación de la ladera, θ = incidencia elipsoidal, φ_sensor = azimut terreno→sensor.
    Ladera mirando al sensor -> θ_local ≈ θ − s (brillante). De espaldas -> θ_local ≈ θ + s (oscura; > 90° = sombra)."""
    s, a, t = np.radians(pendiente), np.radians(orientacion), np.radians(incidencia)
    coseno = np.cos(s) * np.cos(t) + np.sin(s) * np.sin(t) * np.cos(a - np.radians(azimut_sensor))
    return np.degrees(np.arccos(np.clip(coseno, -1, 1)))


def relieve_y_geometria(ee, id_escena_s1, transformacion) -> dict:
    dem = ee.Image("USGS/SRTMGL1_003")
    imagen = (ee.Image.cat([ee.Terrain.slope(dem).rename("pendiente"), ee.Terrain.aspect(dem).rename("orientacion"),
                            ee.Image(f"COPERNICUS/S1_GRD/{id_escena_s1}").select("angle").rename("incidencia")]))
    capas = descargar_ee(ee, imagen, ["pendiente", "orientacion", "incidencia"], transformacion)
    capas["incidencia_local"] = angulo_incidencia_local(capas["pendiente"], capas["orientacion"], capas["incidencia"])
    return capas


# ----------------------------------------------------------------------------- óptico y agua permanente
def estirar(rgb, p_bajo=2, p_alto=98):
    """Estiramiento por percentiles de cada banda -> [0, 1] (solo para visualizar)."""
    salida = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        banda = rgb[..., i]
        validos = banda[np.isfinite(banda)]
        if validos.size == 0:
            continue
        bajo, alto = np.percentile(validos, [p_bajo, p_alto])
        salida[..., i] = np.clip((banda - bajo) / max(alto - bajo, 1e-6), 0, 1)
    return np.nan_to_num(salida)


def s2_compuesto_rgb(ee, transformacion, inicio="2017-04-01", fin="2017-12-31", nubes_max=20):
    """Mediana de Sentinel-2 (L1C) sin nubes en el periodo -> (H, W, 3) en color real, estirado."""
    coleccion = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(_region(ee, transformacion))
                 .filterDate(inicio, fin).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", nubes_max)))
    capas = descargar_ee(ee, coleccion.median(), ["B4", "B3", "B2"], transformacion)
    return estirar(np.stack([capas["B4"], capas["B3"], capas["B2"]], axis=-1))


def jrc_ocurrencia(ee, transformacion):
    """% del tiempo (1984–2021) en que el píxel tuvo agua según JRC Global Surface Water. 0 = nunca."""
    for version in ["JRC/GSW1_4/GlobalSurfaceWater", "JRC/GSW1_3/GlobalSurfaceWater"]:
        try:
            capa = descargar_ee(ee, ee.Image(version).select("occurrence"), ["occurrence"], transformacion)["occurrence"]
            return np.nan_to_num(capa, nan=0.0)
        except Exception:
            continue
    raise RuntimeError("No se pudo leer JRC Global Surface Water")


def s1_en_fecha(ee, id_escena_s1, transformacion):
    capas = descargar_ee(ee, ee.Image(f"COPERNICUS/S1_GRD/{id_escena_s1}"), ["VV", "VH"], transformacion)
    return np.stack([capas["VV"], capas["VH"]])


# ----------------------------------------------------------------------------- Sentinel-2 de Sen1Floods11
def descargar_s2_sen1floods11(ids_chips):
    """Baja SOLO los chips pedidos de S2Hand (cada uno ~7 MB), no los 446."""
    DIR_S2_MANO.mkdir(parents=True, exist_ok=True)
    for id_chip in ids_chips:
        destino = DIR_S2_MANO / f"{id_chip}_S2Hand.tif"
        if not destino.exists():
            subprocess.run(["gsutil", "-q", "cp",
                            f"{BUCKET}/data/flood_events/HandLabeled/S2Hand/{id_chip}_S2Hand.tif", str(destino)],
                           check=True)


def leer_s2_rgb(id_chip):
    """S2Hand trae 13 bandas (B1..B12 con B8A); color real = B4, B3, B2 -> índices 3, 2, 1."""
    with rasterio.open(DIR_S2_MANO / f"{id_chip}_S2Hand.tif") as fuente:
        bandas = fuente.read([4, 3, 2]).astype(np.float32)
    bandas[bandas <= 0] = np.nan
    return estirar(np.moveaxis(bandas, 0, -1))


# ----------------------------------------------------------------------------- cuantificación
def enriquecimiento(mascara_error, condicion, validos) -> dict:
    """¿Los errores se concentran donde se cumple la condición más de lo esperable por azar?
    razón > 1 -> la condición está sobrerrepresentada entre los errores."""
    errores = mascara_error & validos
    pct_errores = 100 * (errores & condicion).sum() / max(errores.sum(), 1)
    pct_base = 100 * (validos & condicion).sum() / max(validos.sum(), 1)
    return {"px_error": int(errores.sum()), "%_errores_con_condicion": round(float(pct_errores), 1),
            "%_area_con_condicion": round(float(pct_base), 1),
            "razon_enriquecimiento": round(float(pct_errores / max(pct_base, 1e-6)), 2)}
