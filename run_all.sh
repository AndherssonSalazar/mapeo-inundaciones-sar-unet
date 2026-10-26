#!/usr/bin/env bash
# =====================================================================
# Comando único: reproduce los resultados de la Pregunta 2 de extremo a extremo.
#   bash run_all.sh                 corrida completa (protocolo del informe: 3 semillas, 60 épocas)
#   bash run_all.sh --quick         prueba rápida (1 semilla, 2 épocas, pocos chips débiles)
#   bash run_all.sh --sin-debiles   omite la ablación de etiquetas débiles (ahorra ~9 GB de descarga)
# Las opciones se pueden combinar. Variables opcionales: RNAP_DATOS (carpeta de datos).
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

for argumento in "$@"; do
  case "$argumento" in
    --quick)        export RNAP_RAPIDO=1 ;;
    --sin-debiles)  export RNAP_SIN_DEBILES=1 ;;
    *) echo "Opción desconocida: $argumento (usa --quick y/o --sin-debiles)"; exit 1 ;;
  esac
done

echo ">> Instalando dependencias (requirements.txt)"
python -m pip install -q -r requirements.txt

inicio=$(date +%s)
for paso in scripts/0*.py; do
  python "$paso"
done
echo ">> Pipeline terminado en $(( ($(date +%s) - inicio) / 60 )) min"
