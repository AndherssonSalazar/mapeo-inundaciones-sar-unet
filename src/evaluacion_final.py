"""
evaluacion_final.py
Fase 7 (entregable 3): evaluación EN DISTRIBUCIÓN sobre la partición test (eventos vistos en
entrenamiento, chips nunca vistos). Reglas:
  - El modelo final se elige SOLO con validación (ablación a): mayor IoU medio sobre 3 semillas.
  - Se reportan las 3 semillas de esa variante: media ± desviación en test.
  - Umbral: 0.5 (fijado a priori) y, como análisis adicional, el umbral que maximiza el IoU en
    VALIDACIÓN aplicado después a test (nunca se ajusta nada mirando test).
"""
import numpy as np
import pandas as pd
import torch

from .entrenamiento import cargar_mejor_modelo
from .metricas import metricas_desde_conteos
from .metricas_torch import AcumuladorConfusion

CARPETAS = {"solo_dorados": "combinada", "solo_debiles": "preentreno",
            "mezcla": "mezcla", "preentreno_ajuste": "preentreno_ajuste"}
N_CASILLAS = 200   # resolución del histograma de probabilidades (umbral cada 0.005)


def elegir_variante(ruta_csv_ablacion) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(ruta_csv_ablacion)
    medias = df.groupby("variante")[["iou", "f1", "precision", "recall"]].agg(["mean", "std"])
    ganadora = medias[("iou", "mean")].idxmax()
    return ganadora, medias.sort_values(("iou", "mean"), ascending=False)


@torch.no_grad()
def evaluar_con_histograma(modelo, cargador, dispositivo):
    """Una sola pasada: acumulador (umbral 0.5) + histogramas de P(agua) para agua y no agua.
    Con los histogramas se puede calcular VP/FP/FN para CUALQUIER umbral sin volver a predecir."""
    modelo.eval()
    acumulador = AcumuladorConfusion(0.5)
    hist_agua = torch.zeros(N_CASILLAS, dtype=torch.float64)
    hist_no_agua = torch.zeros(N_CASILLAS, dtype=torch.float64)
    usar_amp = dispositivo.type == "cuda"
    for imagenes, etiquetas, ids in cargador:
        imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)
        with torch.autocast(dispositivo.type, dtype=torch.float16, enabled=usar_amp):
            logits = modelo(imagenes)
        acumulador.actualizar(logits, etiquetas, ids)
        p = torch.sigmoid(logits.float()).squeeze(1)
        for clase, hist in [(1, hist_agua), (0, hist_no_agua)]:
            valores = p[etiquetas == clase]
            hist += torch.histc(valores, bins=N_CASILLAS, min=0, max=1).double().cpu()
    return acumulador, hist_agua.numpy(), hist_no_agua.numpy()


def curva_umbral(hist_agua, hist_no_agua) -> pd.DataFrame:
    """Métricas para cada umbral u: predice agua si P > u (casillas por encima de u)."""
    filas = []
    total_agua, total_no_agua = hist_agua.sum(), hist_no_agua.sum()
    for i in range(1, N_CASILLAS):
        umbral = i / N_CASILLAS
        vp = hist_agua[i:].sum(); fp = hist_no_agua[i:].sum()
        filas.append({"umbral": umbral, **metricas_desde_conteos(vp, fp, total_agua - vp, total_no_agua - fp)})
    return pd.DataFrame(filas)


def evaluar_semillas(carpeta, semillas, cargadores, dir_experimentos, dispositivo) -> dict:
    """Para cada semilla: métricas en test a 0.5, umbral óptimo en valid y métricas en test con ese umbral."""
    resultado = {"filas": [], "acumuladores_test": {}, "curvas_valid": {}, "curvas_test": {}}
    for semilla in semillas:
        modelo, _, _ = cargar_mejor_modelo(dir_experimentos / f"{carpeta}_s{semilla}", dispositivo)
        _, ha_v, hn_v = evaluar_con_histograma(modelo, cargadores["valid"], dispositivo)
        acum_t, ha_t, hn_t = evaluar_con_histograma(modelo, cargadores["test"], dispositivo)
        curva_v, curva_t = curva_umbral(ha_v, hn_v), curva_umbral(ha_t, hn_t)
        umbral_opt = float(curva_v.loc[curva_v["iou"].idxmax(), "umbral"])
        en_test_opt = curva_t.iloc[(curva_t["umbral"] - umbral_opt).abs().idxmin()]
        resultado["filas"].append({"semilla": semilla, "criterio_umbral": "fijo 0.5", "umbral": 0.5, **acum_t.metricas()})
        resultado["filas"].append({"semilla": semilla, "criterio_umbral": "óptimo en valid", "umbral": umbral_opt,
                                   **en_test_opt.drop("umbral").to_dict()})
        resultado["acumuladores_test"][semilla] = acum_t
        resultado["curvas_valid"][semilla], resultado["curvas_test"][semilla] = curva_v, curva_t
        del modelo
    resultado["df"] = pd.DataFrame(resultado["filas"])
    return resultado


def matriz_sumada(acumuladores) -> np.ndarray:
    return sum(a.matriz_confusion() for a in acumuladores.values())


def por_evento_test(acumuladores) -> pd.DataFrame:
    tablas = [a.tabla_por_evento()[["iou", "f1", "omision", "comision"]] for a in acumuladores.values()]
    return pd.concat(tablas).groupby(level=0).mean().sort_values("iou")
