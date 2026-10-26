"""Paso 7 — E4: generalización geográfica sin reentrenar. Bolivia (evento no visto de Sen1Floods11)
y Lima 2017 (S1 del 17/03/2017 + etiqueta EMSN038 de Copernicus EMS).

Los 64 recortes de Lima se generaron con Google Earth Engine (cuadernos 08a/08b), que exige iniciar
sesión. Para que este paso no dependa de eso, se descargan ya listos del release del repositorio.
Si no se pueden descargar, Lima se omite con un aviso y Bolivia se evalúa igual."""
import io
import urllib.request
import zipfile

import pandas as pd

import _entorno as E
from src.cargadores import crear_cargadores
from src.configuracion import DIR_RESULTADOS, RAIZ_PROYECTO, SEMILLA
from src.evaluacion_final import CARPETAS, elegir_variante
from src.utilidades import obtener_dispositivo

E.titulo("Paso 7/9 — Generalización geográfica (E4)")
dir_lima = RAIZ_PROYECTO / "datos_lima" / "chips"
if not (dir_lima / "S1Hand").exists():
    try:
        print(f"Descargando recortes de Lima: {E.URL_DATOS_LIMA}")
        with urllib.request.urlopen(E.URL_DATOS_LIMA, timeout=120) as respuesta:
            zipfile.ZipFile(io.BytesIO(respuesta.read())).extractall(RAIZ_PROYECTO / "datos_lima")
    except Exception as error:      # sin Internet o release sin el archivo
        print(f"AVISO: no se pudieron descargar los recortes de Lima ({error}). Se evalúa solo Bolivia.")
hay_lima = (dir_lima / "S1Hand").exists() and any((dir_lima / "S1Hand").glob("*.tif"))

dispositivo = obtener_dispositivo()
normalizador = E.normalizador()
cargadores = {"bolivia": crear_cargadores(normalizador, tamano_lote=4, tamano_recorte=256,
                                          num_workers=E.PARAMETROS["num_workers"], semilla=SEMILLA)["bolivia"]}
otsu = pd.read_csv(DIR_RESULTADOS / "f4_linea_base_otsu.csv", index_col=0)["iou"].to_dict()
if hay_lima:
    from src.evaluacion_lima import ConjuntoLima, cargador_lima, ids_lima, otsu_lima
    cargadores["lima"] = cargador_lima(normalizador, num_workers=E.PARAMETROS["num_workers"])
    otsu["lima"] = otsu_lima(ConjuntoLima(ids_lima(), normalizador)).metricas()["iou"]

from src.evaluacion_lima import evaluar_regiones   # noqa: E402

variante, _ = elegir_variante(DIR_RESULTADOS / "f6a_ablacion_corridas.csv")
salida = evaluar_regiones(CARPETAS[variante], E.PARAMETROS["semillas"], cargadores, E.DIR_EXPERIMENTOS, dispositivo)
salida["df"].to_csv(DIR_RESULTADOS / "f8_regiones_por_semilla.csv", index=False)

test = pd.read_csv(DIR_RESULTADOS / "f7_test_por_semilla.csv")
iou_test = test[test["criterio_umbral"] == "fijo 0.5"]["iou"].mean()
filas = [{"region": "test", "iou_unet": iou_test, "iou_otsu": otsu["test"], "brecha_vs_test_%": 0.0}]
for region, grupo in salida["df"].groupby("region"):
    filas.append({"region": region, "iou_unet": grupo["iou"].mean(), "iou_unet_desv": grupo["iou"].std(),
                  "iou_otsu": otsu[region], "brecha_vs_test_%": 100 * (iou_test - grupo["iou"].mean()) / iou_test})
tabla = pd.DataFrame(filas)
tabla.to_csv(DIR_RESULTADOS / "f8_brecha_regiones.csv", index=False)
print(tabla.round(3))
