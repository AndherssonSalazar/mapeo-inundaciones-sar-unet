# Mapeo de inundaciones con Sentinel-1 y U-Net (Sen1Floods11 → Lima 2017)

Código de la **Pregunta 2** del curso *Redes Neuronales y Aprendizaje Profundo* (Maestría en IA, UNI).
U-Net implementada desde cero en PyTorch (sin librerías de segmentación), con entrada VV y VH de
Sentinel-1. Se entrena en [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) y se evalúa
en test, en Bolivia y en la inundación de Lima de marzo de 2017 (Copernicus EMS, EMSN038).

## Resultados principales (IoU de la clase agua, media ± desviación sobre 3 semillas)

| Evaluación | Otsu | U-Net (VV, VH) | Fusión SAR + Sentinel-2 |
|---|---|---|---|
| Test (en distribución) | 0.554 | 0.669 ± 0.003 | 0.848 ± 0.009 |
| Bolivia (evento no visto) | 0.718 | 0.663 ± 0.026 | 0.783 ± 0.008 |
| Lima 2017 (evento no visto) | 0.011 | 0.003 | — |

## Estructura

```
rnap_p2_inundaciones/
├── src/            módulos reutilizables (los escriben los cuadernos con %%writefile)
├── notebooks/      un cuaderno por fase (ver abajo)
├── resultados/     métricas en CSV/JSON y carpetas de cada corrida (experimentos/)
├── figuras/        figuras f1_* ... f10_* usadas en el informe
├── datos_lima/     recortes S1 y etiquetas EMSN038 rasterizadas de Lima
└── informe/        LaTeX (IEEEtran) del informe
```

Módulos clave de `src/`:

| Módulo | Contenido |
|---|---|
| `modelo_unet.py` | `UNet` (DobleConv, Bajada, Subida), conteo de parámetros y campo receptivo empírico |
| `perdidas.py` | BCE, Dice y combinada, todas enmascaradas en los píxeles −1 |
| `metricas_torch.py` | `AcumuladorConfusion`: IoU, F1, precisión, recall, omisión y comisión con conteos agregados |
| `conjunto_datos.py`, `cargadores.py`, `preprocesamiento.py`, `aumentos.py` | carga, normalización (solo con train) y aumentos |
| `entrenamiento.py`, `evaluacion.py`, `experimento.py` | AdamW + calentamiento + coseno, AMP, parada temprana y reanudación |
| `ablacion.py`, `ablacion_debil.py`, `paralelo_6a.py` | ablaciones con 3 semillas y reparto entre 2 GPUs |
| `lima*.py`, `evaluacion_lima.py` | escenas S1 de Earth Engine, rasterizado de EMSN038 y evaluación en Lima |
| `analisis_fallos.py` | incidencia local (SRTM), ΔVV multitemporal, JRC y razón de enriquecimiento |
| `fusion_optica.py`, `tarea_e6.py`, `paralelo_e6.py` | fusión temprana con 6 bandas de Sentinel-2 |

## Orden de ejecución

| Cuaderno | Entregable | Entorno |
|---|---|---|
| `00_entorno_y_descarga` | descarga de Sen1Floods11 | Colab (CPU) |
| `01_exploracion` | análisis exploratorio y estadísticas de normalización | Colab (CPU) |
| `02_dataset_dataloader` | dataset, aumentos y cargadores | Colab (CPU) |
| `03_unet` | **E1**: arquitectura, prueba de sobreajuste y campo receptivo | Colab |
| `04_perdidas_metricas` | pérdidas, métricas y línea base de Otsu | Colab (CPU) |
| `05_entrenamiento_base[_kaggle]` | modelo base | GPU T4 |
| `06b_ablacion_*` | **E2(b)**: BCE / Dice / combinada / sin saltos | GPU T4 |
| `06a_ablacion_debiles_*` | **E2(a)**: dorados vs. débiles | GPU T4 (×2 en Kaggle) |
| `07_evaluacion_test` | **E3**: test, matriz de confusión y umbral | Colab |
| `08a/08b/08c_lima_*` | **E4**: escenas y etiquetas de Lima, generalización | Colab + Earth Engine |
| `09_analisis_fallos` | **E5**: 5 tipos de fallo con evidencia cuantitativa | Colab + Earth Engine |
| `10_fusion_optica_*` | **E6**: fusión con Sentinel-2 | GPU T4 (×2 en Kaggle) |

Notas:
- Los cuadernos de Colab montan Drive en `/content/drive/MyDrive/rnap_p2_inundaciones`. Después de volver a correr un `%%writefile`, hay que reiniciar la sesión.
- Las versiones de Kaggle son autocontenidas: escriben todos los módulos y descargan los datos. Se corren con *Save & Run All (Commit)*.
- Todas las ablaciones son **reanudables**: las corridas con `resumen.json` se saltan.
- Earth Engine requiere un proyecto de Google Cloud registrado (uso no comercial).

## Reproducibilidad

- Semillas 42, 7 y 123. La normalización usa solo estadísticas de train. Las variantes se seleccionan en valid y test se evalúa una sola vez con umbral 0.5.
- Requisitos: Python 3.10+, PyTorch 2.x, rasterio, geopandas, shapely, earthengine-api, pandas, matplotlib.

## Citas

- Bonafilia *et al.*, “Sen1Floods11”, CVPRW 2020.
- Ronneberger *et al.*, “U-Net”, MICCAI 2015.
- Copernicus Emergency Management Service, producto EMSN038 (Lima, 2017).
