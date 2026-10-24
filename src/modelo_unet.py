"""
modelo_unet.py
U-Net implementada desde cero (Ronneberger et al., 2015), adaptada a SAR de 2 bandas.

Arquitectura (profundidad=4, canales_base=c):
  Entrada (2, H, W)
  ├─ Codificador  : DobleConv(2→c) → [MaxPool + DobleConv] x4  (c→2c→4c→8c→16c)   "qué hay"
  ├─ Decodificador: [ConvTranspuesta ×2 + concatenar salto + DobleConv] x4         "dónde está"
  └─ Cabeza 1x1   : c → 1 canal = logit de "agua" por píxel

Diferencias con el paper original (documentarlas en el informe):
  - padding=1 ("same"): la salida tiene el mismo tamaño que la entrada (el original recortaba).
  - BatchNorm tras cada convolución (no existía en 2015; estabiliza y acelera el entrenamiento).
  - 1 canal de salida con sigmoide (problema binario) en vez de 2 con softmax.
  - Parámetros configurables para la ablación: profundidad, canales_base y usar_saltos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DobleConv(nn.Module):
    """(Conv 3x3 → BatchNorm → ReLU) x 2. Bloque básico de la U-Net."""

    def __init__(self, canales_entrada: int, canales_salida: int, abandono: float = 0.0):
        super().__init__()
        capas = [
            nn.Conv2d(canales_entrada, canales_salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales_salida, canales_salida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canales_salida),
            nn.ReLU(inplace=True),
        ]
        if abandono > 0:
            capas.append(nn.Dropout2d(abandono))
        self.bloque = nn.Sequential(*capas)

    def forward(self, x):
        return self.bloque(x)


class Bajada(nn.Module):
    """MaxPool 2x2 (reduce resolución a la mitad) + DobleConv (duplica canales)."""

    def __init__(self, canales_entrada: int, canales_salida: int, abandono: float = 0.0):
        super().__init__()
        self.agrupar = nn.MaxPool2d(2)
        self.conv = DobleConv(canales_entrada, canales_salida, abandono)

    def forward(self, x):
        return self.conv(self.agrupar(x))


class Subida(nn.Module):
    """ConvTranspuesta 2x2 (duplica resolución) + concatenar el salto del codificador + DobleConv."""

    def __init__(self, canales_entrada: int, canales_salida: int, usar_salto: bool = True):
        super().__init__()
        self.usar_salto = usar_salto
        self.subir = nn.ConvTranspose2d(canales_entrada, canales_entrada // 2, kernel_size=2, stride=2)
        canales_conv = canales_entrada // 2 + (canales_salida if usar_salto else 0)
        self.conv = DobleConv(canales_conv, canales_salida)

    def forward(self, x, salto):
        x = self.subir(x)
        # si H o W no eran divisibles entre 2^profundidad, ajustamos con padding al tamaño del salto
        dif_alto, dif_ancho = salto.shape[2] - x.shape[2], salto.shape[3] - x.shape[3]
        if dif_alto or dif_ancho:
            x = F.pad(x, [dif_ancho // 2, dif_ancho - dif_ancho // 2, dif_alto // 2, dif_alto - dif_alto // 2])
        if self.usar_salto:
            x = torch.cat([salto, x], dim=1)   # detalle espacial fino (salto) + contexto (x)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, canales_entrada: int = 2, canales_base: int = 32, profundidad: int = 4,
                 usar_saltos: bool = True, abandono: float = 0.0):
        super().__init__()
        self.configuracion = dict(canales_entrada=canales_entrada, canales_base=canales_base,
                                  profundidad=profundidad, usar_saltos=usar_saltos, abandono=abandono)
        canales = [canales_base * 2 ** i for i in range(profundidad + 1)]   # ej. 32, 64, 128, 256, 512

        self.entrada = DobleConv(canales_entrada, canales[0])
        # el abandono (dropout) solo en los niveles más profundos, donde hay más parámetros
        self.codificador = nn.ModuleList(
            Bajada(canales[i], canales[i + 1], abandono if i >= profundidad - 2 else 0.0)
            for i in range(profundidad)
        )
        self.decodificador = nn.ModuleList(
            Subida(canales[i + 1], canales[i], usar_saltos) for i in reversed(range(profundidad))
        )
        self.salida = nn.Conv2d(canales[0], 1, kernel_size=1)
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        """He/Kaiming: pensada para ReLU, mantiene la varianza de las activaciones entre capas."""
        for modulo in self.modules():
            if isinstance(modulo, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
                if modulo.bias is not None:
                    nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.BatchNorm2d):
                nn.init.ones_(modulo.weight)
                nn.init.zeros_(modulo.bias)

    def forward(self, x, devolver_formas: bool = False):
        formas = [("entrada", tuple(x.shape))]
        x = self.entrada(x)
        saltos = [x]
        formas.append(("codificador nivel 0", tuple(x.shape)))
        for nivel, bajada in enumerate(self.codificador, start=1):
            x = bajada(x)
            saltos.append(x)
            formas.append((f"codificador nivel {nivel}" + (" (cuello)" if nivel == len(self.codificador) else ""),
                           tuple(x.shape)))
        saltos.pop()  # el cuello de botella no es salto de nadie
        for nivel, subida in zip(reversed(range(len(saltos))), self.decodificador):
            x = subida(x, saltos.pop())
            formas.append((f"decodificador nivel {nivel}", tuple(x.shape)))
        logits = self.salida(x)
        formas.append(("salida (logits)", tuple(logits.shape)))
        return (logits, formas) if devolver_formas else logits


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def campo_receptivo(modelo: nn.Module, tamano: int = 512) -> int:
    """Campo receptivo EMPÍRICO: cuántos píxeles de la entrada influyen en el píxel central de la
    salida. Se mide con el gradiente: d(salida_centro)/d(entrada) es distinto de 0 solo dentro del campo.
    Indica cuánto 'contexto' mira la red para decidir si un píxel es agua."""
    estaba_entrenando = modelo.training
    modelo.eval()
    parametro = next(modelo.parameters())
    x = torch.randn(1, modelo.configuracion["canales_entrada"], tamano, tamano,
                    device=parametro.device, requires_grad=True)
    modelo(x)[0, 0, tamano // 2, tamano // 2].backward()
    influye = (x.grad.abs().sum(dim=(0, 1)) > 0)
    filas = torch.where(influye.any(dim=1))[0]
    modelo.train(estaba_entrenando)
    return int(filas.max() - filas.min() + 1)
