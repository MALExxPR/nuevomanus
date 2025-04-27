#!/bin/bash

# Script para ejecutar el sistema de trading con machine learning

# Verificar si se está ejecutando con Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 no está instalado"
    exit 1
fi

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: No se pudo crear el entorno virtual"
        exit 1
    fi
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# Verificar si las dependencias están instaladas
echo "Verificando dependencias..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: No se pudieron instalar las dependencias"
    exit 1
fi

# Crear directorios necesarios
echo "Creando directorios necesarios..."
mkdir -p data/crypto
mkdir -p data/forex
mkdir -p data/processed
mkdir -p models/lstm
mkdir -p models/dqn
mkdir -p results/backtesting
mkdir -p logs

# Ejecutar la aplicación
echo "Iniciando la aplicación..."
streamlit run src/ui/app.py
