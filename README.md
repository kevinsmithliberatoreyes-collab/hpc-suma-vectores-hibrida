# Suma de Vectores en Arquitecturas Heterogéneas (OpenMP + CUDA)

Este repositorio contiene la solución práctica para la **Actividad de la Semana 8**, enfocada en el desarrollo de un algoritmo híbrido en C++/CUDA para realizar la suma elemental de dos vectores masivos (N = 1,048,576 elementos).

El objetivo es comparar el rendimiento del paralelismo multinúcleo en CPU (usando **OpenMP**) frente al paralelismo masivo en GPU (usando **NVIDIA CUDA**).

## 🚀 Requisitos Previos

Para compilar y ejecutar este proyecto de forma nativa, es necesario contar con:
* Un compilador compatible con C++ y soporte para **OpenMP** (como `gcc` / `g++`).
* El entorno **NVIDIA CUDA Toolkit** instalado y configurado.
* Una tarjeta gráfica NVIDIA compatible.

## 🛠️ Compilación y Ejecución

Para compilar el código híbrido asegurando que tanto las directivas de OpenMP como el kernel de CUDA se procesen correctamente, ejecuta el siguiente comando en la consola:

```bash
nvcc -O3 -Xcompiler -fopenmp vector_addition.cu -o vector_addition
