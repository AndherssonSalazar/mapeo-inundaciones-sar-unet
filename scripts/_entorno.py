"""
_entorno.py
Entorno común de los scripts del pipeline (lo importa cada paso antes que nada).

Los módulos de src/ se escribieron para los cuadernos de Colab, donde src/configuracion.py apunta a
Google Drive. Para correr fuera de Colab sin tocar ese archivo, aquí se registra una configuración
equivalente con rutas relativas al repositorio ANTES de importar cualquier módulo de src/: todos
hacen `from .configuracion import ...`, así que leen estas rutas.

Rutas (se pueden cambiar con variables de entorno):
  RNAP_DATOS   carpeta de los datos descargados (por defecto /tmp/datos_sen1floods11 en Kaggle,
               datos/ en cualquier otro lado). Pesan varios GB: no van en el repositorio.
  RNAP_RAPIDO  "1" = prueba rápida (la activa run_all.sh --quick).
Las salidas van a salidas/ (corrida completa) o salidas_rapida/ (prueba rápida), separadas para que
la prueba rápida nunca se confunda con la completa: las corridas son reanudables y se saltan si ya
tienen resumen.json.
"""
import os
import sys
import types
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
RAPIDO = os.environ.get("RNAP_RAPIDO") == "1"
EN_KAGGLE = Path("/kaggle").exists()

# Repositorio de GitHub (para descargar los recortes de Lima publicados en el release)
REPOSITORIO = os.environ.get("RNAP_REPOSITORIO", "USUARIO/mapeo-inundaciones-sar-unet")
VERSION = os.environ.get("RNAP_VERSION", "entrega-final")
URL_DATOS_LIMA = f"https://github.com/{REPOSITORIO}/releases/download/{VERSION}/datos_lima.zip"

RAIZ_DATOS = Path(os.environ.get("RNAP_DATOS") or
                  ("/tmp/datos_sen1floods11" if EN_KAGGLE else RAIZ_REPO / "datos"))
RAIZ_SALIDAS = RAIZ_REPO / ("salidas_rapida" if RAPIDO else "salidas")

# ----------------------------------------------------------------------------- configuración
configuracion = types.ModuleType("src.configuracion")
configuracion.__dict__.update(
    RAIZ_PROYECTO=RAIZ_SALIDAS,
    RAIZ_DATOS=RAIZ_DATOS,
    BUCKET="gs://sen1floods11/v1.1",
    DIR_S1_MANO=RAIZ_DATOS / "HandLabeled" / "S1Hand",
    DIR_ETIQUETA_MANO=RAIZ_DATOS / "HandLabeled" / "LabelHand",
    DIR_OTSU_MANO=RAIZ_DATOS / "HandLabeled" / "S1OtsuLabelHand",
    DIR_S2_MANO=RAIZ_DATOS / "HandLabeled" / "S2Hand",
    DIR_S1_DEBIL=RAIZ_DATOS / "WeaklyLabeled" / "S1Weak",
    DIR_ETIQUETA_DEBIL=RAIZ_DATOS / "WeaklyLabeled" / "S1OtsuLabelWeak",
    DIR_PARTICIONES=RAIZ_DATOS / "splits" / "flood_handlabeled",
    DIR_RESULTADOS=RAIZ_SALIDAS / "resultados",
    DIR_CHECKPOINTS=RAIZ_SALIDAS / "checkpoints",
    DIR_FIGURAS=RAIZ_SALIDAS / "figuras",
    VALOR_SIN_DATO=-1,
    TAMANO_CHIP=512,
    SEMILLA=42,
)
sys.path.insert(0, str(RAIZ_REPO))
import src  # noqa: E402

sys.modules["src.configuracion"] = configuracion
src.configuracion = configuracion

for carpeta in [configuracion.DIR_RESULTADOS, configuracion.DIR_FIGURAS]:
    carpeta.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")    # sin pantalla: las figuras solo se guardan en salidas/figuras/

# ----------------------------------------------------------------------------- parámetros
# Corrida completa = exactamente el protocolo del informe. Prueba rápida = mismo código, pocos datos.
PARAMETROS = dict(
    semillas=[42] if RAPIDO else [42, 7, 123],
    epocas=2 if RAPIDO else 60,
    paciencia=2 if RAPIDO else 15,
    tamano_lote=8,
    tamano_recorte=256,
    num_workers=2,
    pixeles_por_chip=500 if RAPIDO else 4000,      # estadísticas de normalización (paso 2)
    debiles_a_descargar=40 if RAPIDO else None,    # None = todos (~9 GB)
    debiles_mezcla=16 if RAPIDO else 750,
    debiles_preentreno=16 if RAPIDO else 1000,
    epocas_preentreno=1 if RAPIDO else 30,
)
DIR_EXPERIMENTOS = configuracion.DIR_RESULTADOS / "experimentos"


def titulo(texto: str) -> None:
    modo = "PRUEBA RÁPIDA (sus métricas no son resultados)" if RAPIDO else "corrida completa"
    print(f"\n{'=' * 78}\n{texto}  [{modo}]\n  datos: {RAIZ_DATOS}\n  salidas: {RAIZ_SALIDAS}\n{'=' * 78}",
          flush=True)


def configuracion_base(**cambios):
    from src.experimento import ConfiguracionExperimento
    return ConfiguracionExperimento(epocas=PARAMETROS["epocas"], paciencia=PARAMETROS["paciencia"], **cambios)


def normalizador():
    from src.preprocesamiento import Normalizador
    return Normalizador.desde_json(configuracion.DIR_RESULTADOS / "f1_estadisticas_normalizacion.json")
