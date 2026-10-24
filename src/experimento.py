"""
experimento.py
Configuración de un experimento. Cada corrida (base y ablaciones) se describe con UNA instancia,
se guarda como JSON junto a sus resultados -> trazabilidad total para el informe.
"""
import json
from dataclasses import dataclass, asdict, field


@dataclass
class ConfiguracionExperimento:
    nombre: str = "base_bce"
    # --- datos ---
    fuente_entrenamiento: str = "mano"      # "mano" | "mano+debil" (fase 6)
    tamano_lote: int = 8
    tamano_recorte: int = 256
    # --- modelo ---
    canales_base: int = 32
    profundidad: int = 4
    usar_saltos: bool = True
    abandono: float = 0.0
    # --- pérdida ---
    perdida: str = "bce"                     # "bce" | "dice" | "combinada"
    parametros_perdida: dict = field(default_factory=dict)
    # --- optimización ---
    tasa_aprendizaje: float = 1e-3
    decaimiento_pesos: float = 1e-4          # regularización L2 desacoplada (AdamW)
    epocas: int = 100
    epocas_calentamiento: int = 3            # la tasa sube linealmente al inicio (evita saltos bruscos)
    paciencia: int = 20                      # parada temprana: épocas sin mejorar IoU de validación
    umbral: float = 0.5
    semilla: int = 42

    def guardar(self, ruta):
        with open(ruta, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta) as f:
            return cls(**json.load(f))
