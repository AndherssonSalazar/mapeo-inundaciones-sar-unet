# Examen Parcial - Pregunta 2 — Mapeo de inundaciones con Sentinel-1 y U-Net

U-Net implementada desde cero en PyTorch (sin librerías de segmentación) que segmenta agua a partir de
las polarizaciones VV y VH de Sentinel-1. Se entrena en Sen1Floods11, se evalúa en test, en el evento
no visto de Bolivia y en la inundación de Lima de marzo de 2017 (Copernicus EMS, EMSN038), y se
compara con una fusión con Sentinel-2 (bono).

## Datos

- **Sen1Floods11** (Bonafilia *et al.*, 2020): bucket público `gs://sen1floods11/v1.1`. El pipeline lo
  descarga solo, de forma anónima (sin cuenta ni credenciales). **No** se versiona en el repositorio.
- **Lima 2017**: 64 recortes de 512×512 en la misma grilla que Sen1Floods11, con la escena Sentinel-1B
  del 17/03/2017 y la etiqueta EMSN038 rasterizada. Se generaron con Google Earth Engine (cuadernos
  `08a` y `08b`), que exige iniciar sesión; para no depender de eso, `run_all.sh` los descarga ya listos
  del release `entrega-final` (`datos_lima.zip`).

## Ejecución (comando único)

**Entorno recomendado: Kaggle Notebook.** Es donde se corrieron los entrenamientos, y tiene tres
ventajas prácticas:

- **GPU T4 gratuita**, necesaria para entrenar.
- **Internet** para descargar Sen1Floods11 desde Google Cloud (~1 GB los chips a mano, ~3 GB el óptico y
  ~9 GB los chips débiles), que se guardan en `/tmp` y no ocupan el espacio de salida del notebook.
- **Sesiones de hasta 12 h**, suficientes para la corrida completa.

También puede ejecutarse en Colab o en una PC con GPU (ver
[Ejecución fuera de Kaggle](#ejecución-fuera-de-kaggle)).

En un **Kaggle Notebook** nuevo:

1. *Settings*: **Accelerator = GPU T4** (x1 o x2), **Internet = On**.
2. Ejecutar:

```bash
!git clone --branch entrega-final https://github.com/USUARIO/mapeo-inundaciones-sar-unet.git
!cd mapeo-inundaciones-sar-unet && bash run_all.sh
```

`--branch entrega-final` descarga exactamente la versión entregada (release `entrega-final`); Git avisa
"detached HEAD", lo cual es normal al clonar un release. `run_all.sh` instala `requirements.txt` y
ejecuta los pasos 1 a 9 en orden. Los resultados quedan en `mapeo-inundaciones-sar-unet/salidas/`
(`resultados/` con las tablas en CSV y `figuras/` con las figuras).

### Prueba rápida (recomendada para verificar)

```bash
!cd mapeo-inundaciones-sar-unet && bash run_all.sh --quick
```

Corre **todos** los pasos de extremo a extremo con **1 semilla**, **2 épocas** por entrenamiento y solo
40 chips débiles. Sirve para comprobar que el pipeline funciona sin esperar las horas de la corrida
completa; **sus métricas no son resultados**. Escribe en `salidas_rapida/`, separada de la corrida
completa, así que no interfiere con ella.

<!-- Tras correr la prueba rápida en Kaggle, anotar aquí la fecha y la duración. -->

### Corrida completa

El mismo comando de arriba, sin `--quick`, reproduce los resultados del informe:

```bash
!cd mapeo-inundaciones-sar-unet && bash run_all.sh
```

Son 3 semillas por configuración y hasta 60 épocas por entrenamiento (30 entrenamientos en total). En
una GPU T4 toma unas **4–5 horas**; lo más costoso son la ablación de etiquetas débiles y la fusión con
Sentinel-2. Para dejarla corriendo en segundo plano en Kaggle, usar **Save Version → Save & Run All
(Commit)**: al terminar, `salidas/` queda en la pestaña *Output* del notebook.

Opción para ahorrar tiempo y disco: `bash run_all.sh --sin-debiles` omite la ablación de etiquetas
débiles (E2a) y su descarga de ~9 GB; el resto del pipeline no cambia.

<details>
<summary>Si la sesión se corta</summary>

Todas las corridas son **reanudables**: cada entrenamiento guarda un checkpoint por época, y una corrida
que ya tiene `resumen.json` se salta. Basta con volver a ejecutar el mismo comando en la misma sesión
para continuar donde quedó. En Kaggle, si la sesión se cerró, los archivos de `/kaggle/working` solo se
conservan si se corrió como *Commit*.
</details>

> Los números de una nueva corrida completa pueden diferir en el tercer decimal de los del informe:
> el entrenamiento en GPU no es 100 % determinista, y en el informe las corridas se repartieron entre
> Colab y Kaggle (2 GPUs en paralelo), mientras que `run_all.sh` las corre en serie en una sola GPU.

## Ejecución fuera de Kaggle

El código no depende de Kaggle. Requisitos: GPU con CUDA, Bash (Linux, macOS o Colab; en Windows usar
WSL), Python 3.10+ con PyTorch instalado para tu versión de CUDA, Internet y unos 15 GB libres de disco.
Este entorno no fue verificado por nosotros; el verificado es Kaggle.

```bash
git clone --branch entrega-final https://github.com/USUARIO/mapeo-inundaciones-sar-unet.git
cd mapeo-inundaciones-sar-unet
bash run_all.sh --quick     # prueba rápida
# bash run_all.sh           # corrida completa
```

Por defecto los datos se descargan en `datos/` dentro del repositorio (en Kaggle, en
`/tmp/datos_sen1floods11`). Para usar otra carpeta: `RNAP_DATOS=/ruta/a/datos bash run_all.sh`.

En **Colab** (GPU T4), en una celda:

```bash
!git clone --branch entrega-final https://github.com/USUARIO/mapeo-inundaciones-sar-unet.git
!cd mapeo-inundaciones-sar-unet && bash run_all.sh --quick
```

## Pipeline

| Paso | Script | Entregable | Qué hace |
|---|---|---|---|
| 1 | `scripts/01_descarga.py` | — | Descarga anónima de Sen1Floods11: chips a mano (S1, etiquetas, Otsu, S2), particiones y chips débiles |
| 2 | `scripts/02_normalizacion.py` | — | Percentiles 1/99, media y desviación de VV y VH, **solo con train** |
| 3 | `scripts/03_linea_base_otsu.py` | — | Umbral de Otsu evaluado con las mismas métricas y máscara que la U-Net |
| 4 | `scripts/04_ablacion_perdida.py` | **E1**, **E2(b)** | U-Net propia (`src/modelo_unet.py`); BCE vs. Dice vs. combinada y sin conexiones de salto |
| 5 | `scripts/05_ablacion_debiles.py` | **E2(a)** | Solo dorados vs. solo débiles, mezcla y preentreno + ajuste |
| 6 | `scripts/06_evaluacion_test.py` | **E3** | Modelo elegido en valid → test: IoU, F1, precisión, recall, matriz de confusión, curva IoU-umbral, por evento |
| 7 | `scripts/07_generalizacion.py` | **E4** | Bolivia y Lima sin reentrenar; brecha respecto a test |
| 8 | `scripts/08_fusion_optica.py` | **E6** (bono) | Fusión temprana con 6 bandas de Sentinel-2; cambio de IoU pareado por semilla |
| 9 | `scripts/09_resumen.py` | — | Tabla final (`salidas/resultados/resumen_final.csv`) |

**E5 (análisis cualitativo de fallos)** no está en el comando único: necesita varias capas de Google
Earth Engine (relieve SRTM, imágenes antes y después del evento, JRC Global Surface Water) y una cuenta
registrada. Se reproduce con el cuaderno `notebooks/09_analisis_fallos.ipynb` en Colab; sus figuras y
su tabla de evidencias están en `figuras/f9_*` y `resultados/f9_resumen_fallos.csv`.

## Estructura

```
run_all.sh          comando único (--quick para la prueba rápida, --sin-debiles opcional)
requirements.txt    dependencias
scripts/            un script por paso del pipeline (01 … 09) y el entorno común (_entorno.py)
src/                código propio: U-Net, pérdidas, métricas, datos, entrenamiento, ablaciones, Lima, fusión
notebooks/          cuadernos de Colab/Kaggle con los que se desarrolló y corrió cada fase (00 … 10)
resultados/         resultados del informe (CSV/JSON por fase y métricas de cada corrida)
figuras/            figuras del informe (f1_* … f10_*)
informe/            informe en LaTeX (IEEEtran)
salidas/            lo que genera run_all.sh (no se versiona)
```

`src/configuracion.py` conserva las rutas de Google Drive que usan los cuadernos de Colab. Los scripts
no lo modifican: `scripts/_entorno.py` registra una configuración equivalente con rutas relativas al
repositorio antes de importar `src/`.

## Resultados principales (IoU de la clase agua, media ± desviación sobre 3 semillas)

| Evaluación | Otsu | U-Net (VV, VH) | Fusión SAR + Sentinel-2 |
|---|---|---|---|
| Test (eventos vistos) | 0.554 | 0.669 ± 0.003 | 0.848 ± 0.009 |
| Bolivia (evento no visto) | 0.718 | 0.663 ± 0.026 | 0.783 ± 0.008 |
| Lima 2017 (evento no visto) | 0.011 | 0.003 | — |

### Ablaciones (valid)

| Variante | IoU | Precisión | Recall |
|---|---|---|---|
| BCE | 0.621 ± 0.002 | 0.844 | 0.702 |
| Dice | 0.634 ± 0.009 | 0.800 | 0.753 |
| **BCE + Dice (modelo final)** | **0.644 ± 0.002** | **0.822** | **0.749** |
| BCE sin conexiones de salto | 0.584 ± 0.007 | 0.773 | 0.706 |
| Solo débiles | 0.562 ± 0.009 | 0.731 | 0.709 |
| Mezcla dorados + débiles | 0.571 ± 0.004 | 0.751 | 0.706 |
| Preentreno con débiles + ajuste | 0.641 ± 0.005 | 0.823 | 0.744 |

## Reproducibilidad

- Semillas 42, 7 y 123. Normalización calculada solo con train. Las variantes se comparan en valid y
  test se evalúa una sola vez con el modelo elegido y umbral 0.5 fijado de antemano.
- Los pesos de cada corrida se guardan en `salidas/resultados/experimentos/<corrida>/mejor.pt`.

## Citas

- Bonafilia *et al.*, “Sen1Floods11: A georeferenced dataset to train and test deep learning flood
  algorithms for Sentinel-1”, CVPRW 2020.
- Ronneberger *et al.*, “U-Net: Convolutional networks for biomedical image segmentation”, MICCAI 2015.
- Copernicus Emergency Management Service, producto EMSN038 (Lima, 2017).
