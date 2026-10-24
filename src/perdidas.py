"""
perdidas.py
Funciones de pérdida para segmentación binaria con píxeles a IGNORAR (etiqueta = -1).
Todas reciben:
  logits    : (B, 1, H, W) o (B, H, W)  -> salida cruda de la U-Net (sin sigmoide)
  etiquetas : (B, H, W) int64           -> 0 no agua, 1 agua, -1 ignorar
Las tres opciones son las de la ablación (b): BCE vs Dice vs combinada.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VALOR_IGNORAR = -1


def _aplanar_validos(logits, etiquetas):
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    validos = etiquetas != VALOR_IGNORAR
    return logits[validos], (etiquetas[validos] == 1).float()


class PerdidaBCE(nn.Module):
    """Entropía cruzada binaria por píxel.
    + Gradientes estables y bien comportados desde el inicio.
    - Cada píxel pesa igual: con ~9 % de agua, la red puede 'ganar' prediciendo casi todo no-agua.
    peso_positivo > 1 compensa ese desbalance multiplicando el error en píxeles de agua."""

    def __init__(self, peso_positivo: float | None = None):
        super().__init__()
        self.peso_positivo = peso_positivo

    def forward(self, logits, etiquetas):
        x, y = _aplanar_validos(logits, etiquetas)
        if x.numel() == 0:                       # lote sin píxeles válidos: pérdida 0 sin romper el grafo
            return logits.sum() * 0.0
        peso = None if self.peso_positivo is None else torch.tensor(self.peso_positivo, device=x.device)
        return F.binary_cross_entropy_with_logits(x, y, pos_weight=peso)


class PerdidaDice(nn.Module):
    """Pérdida Dice 'suave' (Milletari et al., 2016):  1 - 2·Σ(p·y) / (Σp + Σy)
    con p = sigmoide(logits) (probabilidades continuas, para que sea derivable).
    + Mide superposición de la CLASE AGUA: el fondo abundante no la domina -> robusta al desbalance.
    - Gradientes más ruidosos, sobre todo si el lote tiene poca agua.
    Se calcula sobre TODO el lote (no chip por chip): así un chip sin agua no produce 0/0."""

    def __init__(self, suavizado: float = 1.0):
        super().__init__()
        self.suavizado = suavizado

    def forward(self, logits, etiquetas):
        x, y = _aplanar_validos(logits, etiquetas)
        if x.numel() == 0:
            return logits.sum() * 0.0
        p = torch.sigmoid(x.float())             # float32 aunque se entrene con AMP (sumas grandes)
        interseccion = (p * y).sum()
        return 1 - (2 * interseccion + self.suavizado) / (p.sum() + y.sum() + self.suavizado)


class PerdidaCombinada(nn.Module):
    """peso_bce · BCE + peso_dice · Dice: estabilidad de la BCE + foco en la clase minoritaria del Dice."""

    def __init__(self, peso_bce: float = 0.5, peso_dice: float = 0.5, peso_positivo: float | None = None):
        super().__init__()
        self.peso_bce, self.peso_dice = peso_bce, peso_dice
        self.bce = PerdidaBCE(peso_positivo)
        self.dice = PerdidaDice()

    def forward(self, logits, etiquetas):
        return self.peso_bce * self.bce(logits, etiquetas) + self.peso_dice * self.dice(logits, etiquetas)


def crear_perdida(nombre: str, **parametros) -> nn.Module:
    """Fábrica para la ablación: crear_perdida('bce' | 'dice' | 'combinada')."""
    opciones = {"bce": PerdidaBCE, "dice": PerdidaDice, "combinada": PerdidaCombinada}
    if nombre not in opciones:
        raise ValueError(f"Pérdida desconocida '{nombre}'. Opciones: {list(opciones)}")
    return opciones[nombre](**parametros)
