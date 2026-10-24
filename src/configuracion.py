"""
configuracion.py
Rutas y constantes globales del proyecto (Pregunta 2 - Sen1Floods11 + U-Net).
Todo lo que sea "dónde están las cosas" vive aquí, para no repetir rutas en cada notebook.
"""
from pathlib import Path

# --- Raíz del proyecto en Google Drive (código, resultados, checkpoints) ---
RAIZ_PROYECTO = Path("/content/drive/MyDrive/rnap_p2_inundaciones")

# --- Datos: en el disco local de Colab (rápido, se re-descarga por sesión) ---
RAIZ_DATOS = Path("/content/datos_sen1floods11")
BUCKET = "gs://sen1floods11/v1.1"

# Chips etiquetados a mano (446)
DIR_S1_MANO        = RAIZ_DATOS / "HandLabeled" / "S1Hand"           # entrada SAR VV/VH (dB)
DIR_ETIQUETA_MANO  = RAIZ_DATOS / "HandLabeled" / "LabelHand"        # etiqueta dorada
DIR_OTSU_MANO      = RAIZ_DATOS / "HandLabeled" / "S1OtsuLabelHand"  # etiqueta Otsu sobre los MISMOS chips (para medir ruido)
DIR_S2_MANO        = RAIZ_DATOS / "HandLabeled" / "S2Hand"           # óptico (solo para el bono, fase 10)

# Chips con etiqueta débil (4,384) -> se descargan recién en la fase 6
DIR_S1_DEBIL       = RAIZ_DATOS / "WeaklyLabeled" / "S1Weak"
DIR_ETIQUETA_DEBIL = RAIZ_DATOS / "WeaklyLabeled" / "S1OtsuLabelWeak"

# Particiones oficiales
DIR_PARTICIONES = RAIZ_DATOS / "splits" / "flood_handlabeled"

# --- Salidas ---
DIR_RESULTADOS  = RAIZ_PROYECTO / "resultados"
DIR_CHECKPOINTS = RAIZ_PROYECTO / "checkpoints"
DIR_FIGURAS     = RAIZ_PROYECTO / "figuras"   # irán al informe LaTeX

# --- Convenciones del dataset ---
VALOR_SIN_DATO = -1      # en las etiquetas: -1 sin dato, 0 no agua, 1 agua
TAMANO_CHIP = 512
SEMILLA = 42
