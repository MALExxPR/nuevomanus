#!/bin/bash

# Script para generar el ejecutable del sistema de trading
# Requiere Python 3 y pip instalados en el sistema

set -e

if ! command -v python3 >/dev/null; then
    echo "Error: Python 3 no está instalado" >&2
    exit 1
fi

# Crear y activar entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi
source venv/bin/activate

# Instalar dependencias y PyInstaller
pip install -r requirements.txt
pip install pyinstaller

# Generar ejecutable con PyInstaller
pyinstaller TradingML.spec

echo "Ejecutable creado en el directorio dist/"
