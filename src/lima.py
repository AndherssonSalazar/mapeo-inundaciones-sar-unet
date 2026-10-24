"""
lima.py
Fase 8: región de prueba de Lima (marzo 2017, El Niño costero).
  - Sentinel-1 GRD desde Google Earth Engine (COPERNICUS/S1_GRD): mismo preprocesamiento que
    Sen1Floods11 (calibración σ0, corrección de terreno, VV/VH en dB, 10 m).
  - Delineación de inundación de Copernicus EMS, activación EMSN038 (WorldView-2, 18 y 21/03/2017).
"""
import zipfile
from pathlib import Path

import pandas as pd
import requests

# Rectángulo de trabajo (lon_min, lat_min, lon_max, lat_max): cuencas bajas de Rímac y Chillón,
# de Carapongo/Cajamarquilla (este) a Puente Piedra/San Martín de Porres (oeste).
# Se refina después con el AOI real del producto EMS.
RECTANGULO_LIMA = [-77.15, -12.08, -76.78, -11.78]

URL_EMSN038 = "https://mapping.emergency.copernicus.eu/download/EMSN038/EMSN038_Geodatabase.zip"


# ----------------------------------------------------------------------------- Earth Engine
def buscar_escenas_s1(ee, rectangulo=RECTANGULO_LIMA, inicio="2017-02-15", fin="2017-04-15") -> pd.DataFrame:
    """Lista las escenas Sentinel-1 IW que tocan el rectángulo: fecha, satélite, órbita y cobertura."""
    region = ee.Geometry.Rectangle(rectangulo)
    coleccion = (ee.ImageCollection("COPERNICUS/S1_GRD")
                 .filterBounds(region).filterDate(inicio, fin)
                 .filter(ee.Filter.eq("instrumentMode", "IW")))

    def describir(imagen):
        interseccion = imagen.geometry().intersection(region, 100).area(100)
        return ee.Feature(None, {
            "id": imagen.id(),
            "fecha_hora": imagen.date().format("YYYY-MM-dd HH:mm"),
            "satelite": imagen.get("platform_number"),
            "pasada": imagen.get("orbitProperties_pass"),
            "orbita_relativa": imagen.get("relativeOrbitNumber_start"),
            "polarizaciones": imagen.get("transmitterReceiverPolarisation"),
            "cobertura_%": interseccion.divide(region.area(100)).multiply(100),
        })

    filas = [f["properties"] for f in coleccion.map(describir).getInfo()["features"]]
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["polarizaciones"] = df["polarizaciones"].apply(lambda p: "+".join(p) if isinstance(p, list) else p)
    df["cobertura_%"] = df["cobertura_%"].round(1)
    return df.sort_values("fecha_hora").reset_index(drop=True)


def url_miniatura(ee, id_imagen, banda="VV", rectangulo=RECTANGULO_LIMA, rango=(-25, 0)):
    imagen = ee.Image(f"COPERNICUS/S1_GRD/{id_imagen}").select(banda)
    return imagen.getThumbURL({"region": ee.Geometry.Rectangle(rectangulo), "dimensions": 768,
                               "min": rango[0], "max": rango[1], "format": "png"})


# ----------------------------------------------------------------------------- Copernicus EMS
def descargar_emsn038(dir_destino: Path, url: str = URL_EMSN038) -> Path:
    """Descarga y descomprime el producto vectorial EMSN038. Devuelve la carpeta descomprimida."""
    dir_destino.mkdir(parents=True, exist_ok=True)
    ruta_zip = dir_destino / Path(url).name
    if not ruta_zip.exists():
        respuesta = requests.get(url, timeout=120)
        respuesta.raise_for_status()
        ruta_zip.write_bytes(respuesta.content)
    dir_extraido = dir_destino / ruta_zip.stem
    if not dir_extraido.exists():
        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(dir_extraido)
    # a veces el zip trae otros zips adentro
    for interno in list(dir_extraido.rglob("*.zip")):
        destino = interno.with_suffix("")
        if not destino.exists():
            with zipfile.ZipFile(interno) as z:
                z.extractall(destino)
    return dir_extraido


def listar_capas_vectoriales(dir_raiz: Path) -> pd.DataFrame:
    """Inventario de todas las capas vectoriales (geodatabase, geojson, shp): nombre, geometría, n° de objetos, CRS, columnas."""
    import geopandas as gpd
    import pyogrio

    fuentes = [f for patron in ("*.gdb", "*.gpkg", "*.geojson", "*.json", "*.shp") for f in dir_raiz.rglob(patron)]
    filas = []
    for fuente in fuentes:
        try:
            capas = pyogrio.list_layers(fuente)
        except Exception:
            continue   # p. ej. un .json de metadatos que no es geográfico
        for nombre, tipo in capas:
            try:
                gdf = gpd.read_file(fuente, layer=nombre)
                filas.append({"fuente": str(fuente.relative_to(dir_raiz)), "capa": nombre, "geometria": tipo,
                              "objetos": len(gdf), "crs": str(gdf.crs),
                              "columnas": ", ".join(c for c in gdf.columns if c != "geometry")})
            except Exception as error:
                filas.append({"fuente": str(fuente.relative_to(dir_raiz)), "capa": nombre, "geometria": tipo,
                              "objetos": f"error: {error}"})
    return pd.DataFrame(filas)


def leer_capa(dir_raiz: Path, fuente_relativa: str, capa: str):
    import geopandas as gpd
    return gpd.read_file(dir_raiz / fuente_relativa, layer=capa)
