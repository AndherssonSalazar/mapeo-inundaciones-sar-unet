"""
entrenamiento.py
Bucle de entrenamiento con:
  - AdamW + calentamiento lineal + decaimiento coseno de la tasa de aprendizaje
  - precisión mixta (AMP) en GPU
  - selección del mejor modelo por IoU de VALIDACIÓN (test no se mira hasta el final)
  - parada temprana
  - checkpoint en Drive en cada época -> si Colab se desconecta, se REANUDA donde quedó
"""
import json
import math
import time

import pandas as pd
import torch

from .evaluacion import evaluar
from .modelo_unet import UNet
from .perdidas import crear_perdida
from .utilidades import fijar_semilla


def crear_modelo(configuracion):
    return UNet(canales_entrada=2, canales_base=configuracion.canales_base,
                profundidad=configuracion.profundidad, usar_saltos=configuracion.usar_saltos,
                abandono=configuracion.abandono)


def _factor_tasa(epoca, configuracion):
    """Multiplicador de la tasa de aprendizaje en cada época (0-indexada)."""
    if epoca < configuracion.epocas_calentamiento:
        return (epoca + 1) / configuracion.epocas_calentamiento
    progreso = (epoca - configuracion.epocas_calentamiento) / max(1, configuracion.epocas - configuracion.epocas_calentamiento)
    return 0.5 * (1 + math.cos(math.pi * progreso))


def entrenar(configuracion, cargador_entrenamiento, cargador_validacion, dispositivo, dir_salida,
             reanudar: bool = True):
    dir_salida.mkdir(parents=True, exist_ok=True)
    configuracion.guardar(dir_salida / "configuracion.json")
    fijar_semilla(configuracion.semilla)

    modelo = crear_modelo(configuracion).to(dispositivo)
    perdida = crear_perdida(configuracion.perdida, **configuracion.parametros_perdida)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=configuracion.tasa_aprendizaje,
                                    weight_decay=configuracion.decaimiento_pesos)
    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, lambda e: _factor_tasa(e, configuracion))
    usar_amp = dispositivo.type == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=usar_amp)

    epoca_inicial, mejor_iou, epocas_sin_mejora, historial = 0, -1.0, 0, []
    ruta_ultimo, ruta_mejor = dir_salida / "ultimo.pt", dir_salida / "mejor.pt"
    if reanudar and ruta_ultimo.exists():
        estado = torch.load(ruta_ultimo, map_location=dispositivo, weights_only=False)
        modelo.load_state_dict(estado["modelo"]); optimizador.load_state_dict(estado["optimizador"])
        planificador.load_state_dict(estado["planificador"]); escalador.load_state_dict(estado["escalador"])
        epoca_inicial, mejor_iou = estado["epoca"] + 1, estado["mejor_iou"]
        epocas_sin_mejora, historial = estado["epocas_sin_mejora"], estado["historial"]
        print(f"↻ Reanudando desde la época {epoca_inicial} (mejor IoU valid = {mejor_iou:.4f})")

    for epoca in range(epoca_inicial, configuracion.epocas):
        inicio = time.time()
        modelo.train()
        suma_perdida, n_lotes = 0.0, 0
        for imagenes, etiquetas, _ in cargador_entrenamiento:
            imagenes = imagenes.to(dispositivo, non_blocking=True)
            etiquetas = etiquetas.to(dispositivo, non_blocking=True)
            with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=usar_amp):
                logits = modelo(imagenes)
            valor = perdida(logits.float(), etiquetas)   # la pérdida siempre en float32 (más estable)
            optimizador.zero_grad(set_to_none=True)
            escalador.scale(valor).backward()
            escalador.step(optimizador)
            escalador.update()
            suma_perdida += valor.item(); n_lotes += 1
        planificador.step()

        acumulador, perdida_valid = evaluar(modelo, cargador_validacion, dispositivo, perdida, configuracion.umbral)
        m = acumulador.metricas()
        fila = {"epoca": epoca + 1, "tasa": optimizador.param_groups[0]["lr"],
                "perdida_train": suma_perdida / max(n_lotes, 1), "perdida_valid": perdida_valid,
                **{f"{k}_valid": v for k, v in m.items()}, "segundos": time.time() - inicio}
        historial.append(fila)

        mejoro = m["iou"] > mejor_iou
        if mejoro:
            mejor_iou, epocas_sin_mejora = m["iou"], 0
            torch.save({"modelo": modelo.state_dict(), "epoca": epoca + 1, "iou_valid": mejor_iou,
                        "configuracion": configuracion.__dict__}, ruta_mejor)
        else:
            epocas_sin_mejora += 1
        torch.save({"modelo": modelo.state_dict(), "optimizador": optimizador.state_dict(),
                    "planificador": planificador.state_dict(), "escalador": escalador.state_dict(),
                    "epoca": epoca, "mejor_iou": mejor_iou, "epocas_sin_mejora": epocas_sin_mejora,
                    "historial": historial}, ruta_ultimo)
        pd.DataFrame(historial).to_csv(dir_salida / "historial.csv", index=False)

        print(f"época {epoca + 1:3d} | pérdida train {fila['perdida_train']:.4f} valid {perdida_valid:.4f} | "
              f"IoU valid {m['iou']:.4f} | F1 {m['f1']:.4f} | {fila['segundos']:.0f}s" + ("  ★" if mejoro else ""))
        if epocas_sin_mejora >= configuracion.paciencia:
            print(f"Parada temprana: {configuracion.paciencia} épocas sin mejorar.")
            break

    resumen = {"nombre": configuracion.nombre, "mejor_iou_valid": mejor_iou,
               "mejor_epoca": int(pd.DataFrame(historial)["iou_valid"].idxmax()) + 1,
               "epocas_entrenadas": len(historial)}
    with open(dir_salida / "resumen.json", "w") as f:
        json.dump(resumen, f, indent=2)
    return pd.DataFrame(historial), resumen


def cargar_mejor_modelo(dir_salida, dispositivo):
    from .experimento import ConfiguracionExperimento
    configuracion = ConfiguracionExperimento.cargar(dir_salida / "configuracion.json")
    modelo = crear_modelo(configuracion).to(dispositivo)
    estado = torch.load(dir_salida / "mejor.pt", map_location=dispositivo, weights_only=False)
    modelo.load_state_dict(estado["modelo"])
    modelo.eval()
    return modelo, configuracion, estado["epoca"]
