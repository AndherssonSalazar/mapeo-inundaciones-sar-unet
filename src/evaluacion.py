"""
evaluacion.py
Evaluación de un modelo sobre un DataLoader completo (chips enteros de 512x512).
"""
import torch

from .metricas_torch import AcumuladorConfusion


@torch.no_grad()
def evaluar(modelo, cargador, dispositivo, perdida=None, umbral: float = 0.5):
    """Devuelve (acumulador con métricas globales y por evento, pérdida media o None)."""
    modelo.eval()
    acumulador = AcumuladorConfusion(umbral)
    suma_perdida, n_lotes = 0.0, 0
    usar_amp = dispositivo.type == "cuda"
    for imagenes, etiquetas, ids in cargador:
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        etiquetas = etiquetas.to(dispositivo, non_blocking=True)
        with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=usar_amp):
            logits = modelo(imagenes)
        acumulador.actualizar(logits, etiquetas, ids)
        if perdida is not None:
            suma_perdida += perdida(logits.float(), etiquetas).item()
            n_lotes += 1
    return acumulador, (suma_perdida / n_lotes if n_lotes else None)


@torch.no_grad()
def predecir_probabilidades(modelo, imagen, dispositivo):
    """imagen: tensor (2, H, W) -> mapa de probabilidad de agua (H, W) en CPU."""
    modelo.eval()
    logits = modelo(imagen[None].to(dispositivo))
    return torch.sigmoid(logits.float())[0, 0].cpu()
