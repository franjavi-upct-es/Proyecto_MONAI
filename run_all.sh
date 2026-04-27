#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "No se encontro interprete en $PYTHON_BIN"
    echo "Puedes indicar otro con: PYTHON_BIN=/ruta/a/python ./run_all.sh"
    exit 1
fi

cd "$ROOT_DIR/scripts"

echo "=== Pipeline activo: SegResNet 3D patch-based con MONAI ==="
echo "--- fase1_exploracion ---"
"$PYTHON_BIN" fase1_exploracion.py

echo "--- fase4_segmentacion ---"
"$PYTHON_BIN" fase4_segmentacion.py

echo "--- fase6_escasez ---"
"$PYTHON_BIN" fase6_escasez.py

echo "=== Pipeline 3D completado ==="
echo "Demo opcional: $PYTHON_BIN $ROOT_DIR/scripts/fase7_demo.py"
