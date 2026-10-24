"""
aumentos.py
Aumento de datos geométrico. Regla de oro en segmentación: la MISMA transformación
a la imagen y a la máscara, si no la etiqueta deja de corresponder al píxel.

Solo usamos volteos y rotaciones de 90°: no inventan valores (no hay interpolación),
así que la máscara sigue siendo exactamente {-1, 0, 1}.
Matiz para el informe: el SAR mira de lado (geometría oblicua), por lo que la dirección
de sombras/layover no es invariante a rotaciones; aun así es la práctica estándar
y mejora la generalización con pocos datos (446 chips).
"""
import numpy as np


def aumentar(imagen: np.ndarray, etiqueta: np.ndarray, generador: np.random.Generator):
    """imagen: (C, H, W); etiqueta: (H, W)."""
    if generador.random() < 0.5:                       # volteo horizontal
        imagen, etiqueta = imagen[:, :, ::-1], etiqueta[:, ::-1]
    if generador.random() < 0.5:                       # volteo vertical
        imagen, etiqueta = imagen[:, ::-1, :], etiqueta[::-1, :]
    k = int(generador.integers(0, 4))                  # rotación 0/90/180/270
    if k:
        imagen = np.rot90(imagen, k, axes=(1, 2))
        etiqueta = np.rot90(etiqueta, k, axes=(0, 1))
    return np.ascontiguousarray(imagen), np.ascontiguousarray(etiqueta)


def recorte_aleatorio(imagen: np.ndarray, etiqueta: np.ndarray, tamano: int, generador: np.random.Generator):
    """Recorta una ventana tamano x tamano en posición aleatoria (otra forma de aumento)."""
    _, alto, ancho = imagen.shape
    fila = int(generador.integers(0, alto - tamano + 1))
    columna = int(generador.integers(0, ancho - tamano + 1))
    return (imagen[:, fila:fila + tamano, columna:columna + tamano],
            etiqueta[fila:fila + tamano, columna:columna + tamano])
