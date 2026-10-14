# Barra de Ventaja Táctica en Tenis Profesional 🎾

[![Open In Colab: Pipeline ML](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/federicosolerup/barra_ventaja_tenis/blob/main/pipeline.ipynb)
[![Open In Colab: Video Manim](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/federicosolerup/barra_ventaja_tenis/blob/main/generar_video_barra_evaluacion.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-HistGradient-orange.svg)](https://xgboost.readthedocs.io/)
[![Overleaf LaTeX](https://img.shields.io/badge/Paper-LaTeX%20Overleaf-teal.svg)](overleaf/ProyectoInvestigacionTemplate.tex)

> **Proyecto de Investigación I — Universidad de Palermo (UP)**  
> **Autor:** Federico Soler  
> **Modelo:** Inferencia de Probabilidad Condicional de Victoria y Barra de Evaluación Táctica Dinámica para Tenis Transmitido.

---

## 📌 Descripción del Proyecto

Este proyecto introduce una **barra de ventaja táctica dinámica (Evaluation Bar)** para transmisiones de tenis profesional, inspirada en los motores de análisis posicional del ajedrez (como Stockfish), pero adaptada a la naturaleza probabilística, continua y no determinista del tenis moderno.

A través del procesamiento del corpus del **Match Charting Project (MCP)** sobre partidos ATP masculinos entre jugadores diestros, se diseñó un modelo **XGBoost con memoria secuencial de segundo orden (, t-1, t-2$)** compuesto por **58 variables explicativas One-Hot**. El modelo infiere tiro a tiro la probabilidad condicional de que el jugador al servicio gane el punto ((\\text{Victoria Sacador})$) en tiempo real.

---

## 🚀 Ejecución Inmediata en Google Colab

Podés ejecutar todo el pipeline de extremo a extremo sin necesidad de instalar librerías ni descargar datasets a tu computadora:

* 🔬 **[Ejecutar Pipeline de Machine Learning (XGBoost)](https://colab.research.google.com/github/federicosolerup/barra_ventaja_tenis/blob/main/pipeline.ipynb)**: Carga el modelo pre-entrenado, ejecuta la ablación temporal y genera la simulación gráfica del caso de estudio en **~3 segundos**.
* 🎬 **[Ejecutar Renderizador de Video (Manim)](https://colab.research.google.com/github/federicosolerup/barra_ventaja_tenis/blob/main/generar_video_barra_evaluacion.ipynb)**: Instala Manim automáticamente y renderiza la animación cinematográfica a 1080p 60fps con la barra dinámica sincronizada al peloteo.

---

## 📊 Caso de Estudio: Juan Martín del Potro vs. Roger Federer

Simulación paso a paso de la evolución de probabilidad de victoria en un punto bisagra (Final US Open 2009, 5to set, 4-2, 30-40, Break Point):

![Evolución de la Probabilidad Condicional de Victoria](overleaf/eval_bar_simulation.png)

### Evolución Táctica Tiro a Tiro:
| Paso | Descripción del Tiro | Ejecutor | (\\text{Del Potro})$ | (\\text{Federer})$ | $\\Delta P(\\text{Sacador})$ |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Estado previo al saque | Del Potro | 64.0% | 36.0% | .00$ |
| **1** | 1er Saque falta en red (DP) | Del Potro | 51.0% | 49.0% | $-0.13$ |
| **2** | 2do Saque a la T (DP) | Del Potro | 67.0% | 33.0% | $+0.16$ |
| **3** | Slice de revés al centro profundo (F) | Federer | 57.0% | 43.0% | $-0.10$ |
| **4** | Derecha al centro corta (DP) | Del Potro | 55.0% | 45.0% | $-0.02$ |
| **5** | Revés paralelo profundo (F) | Federer | 43.0% | 57.0% | $-0.12$ |
| **6** | Derecha al centro corta (DP) | Del Potro | 43.0% | 57.0% | .00$ |
| **7** | Approach derecha cruzada media (F) | Federer | 29.0% | 71.0% | $-0.14$ |
| **8** | Derecha paralela media (DP) | Del Potro | 62.0% | 38.0% | $+0.33$ |
| **9** | Volea de revés cruzada corta ganadora (F) | Federer | 0.0% | 100.0% | $-0.62$ |

---

## 🔬 Resultados del Estudio de Ablación Temporal

Evaluación rigurosa sobre partición por grupos de partidos (GroupShuffleSplit 80/20, libre de fuga de datos):

| Modelo | Memoria Temporal | Variables | Accuracy | AUC-ROC | Brier Score | Log Loss | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Uniforme** | Ninguna (=0.50$) | 0 | 50.00% | 50.00% | 0.2500 | 0.6931 | 6.70% |
| **Dummy Mayoría** | Frecuentista (=0.567$) | 0 | 56.70% | 50.00% | 0.2455 | 0.6841 | 0.00% |
| **XGBoost ($)** | Impacto actual | 20 | 66.86% | 72.86% | 0.2106 | 0.6090 | 1.84% |
| **XGBoost (, t-1$)** | Impacto + Respuesta rival | 39 | 67.31% | 73.57% | 0.2078 | 0.6027 | 1.48% |
| **XGBoost (, t-1, t-2$)** | Impacto + Respuesta + Preparación | **58** | **67.65%** | **74.07%** | **0.2057** | **0.5982** | **1.21%** |

---

## 📁 Estructura del Repositorio

`	ext
├── pipeline.ipynb                      # Cuaderno Jupyter reproducible de Machine Learning
├── generar_video_barra_evaluacion.ipynb # Generador cinemático de video con Manim
├── HTML/                               # Interfaz web interactiva del prototipo
│   ├── index.html                      # Layout de simulación con video y barra de ventaja
│   ├── style.css                       # Sistema de diseño, temas visuales y responsive
│   ├── app.js                          # Motor de sincronización, video player y eventos
│   └── model-data.js                   # Simulación del modelo cliente en JavaScript
├── overleaf/                           # Manuscrito formal del proyecto de investigación
│   ├── ProyectoInvestigacionTemplate.tex # Código fuente en LaTeX del artículo
│   ├── Bibliografia.bib                # Base de datos bibliográfica en BibTeX
│   ├── modelo_xgboost_3tiros.json      # Pesos serializados del modelo XGBoost final
│   └── eval_bar_simulation.png        # Gráfico vectorial de alta resolución del caso
├── dataset/                            # Corpus del Match Charting Project (MCP)
│   ├── charting-m-matches.csv          # Metadata de partidos y características de jugadores
│   ├── charting-m-points-modern.csv.zip# Registro comprimido de puntos tiro a tiro
│   └── data_dictionary.txt             # Diccionario taxonómico oficial de tiros MCP
├── Calculo_Metricas_Consolidado.xlsx   # Hojas de cálculo con el desglose de métricas
└── Caso_de_Estudio_XGBoost_Explicado.xlsx # Cálculos matemáticos detallados del caso
`

---

## 💻 Simulación Web Interactiva

El directorio /HTML contiene una aplicación web completa construida con tecnologías estándar (HTML5 Canvas, CSS Vanilla y JavaScript moderno) que reproduce la transmisión de televisión deportiva incorporando la barra lateral interactiva:
* **Modos de visualización**: Tema Oscuro Broadcast y Tema Verde Chroma (para incrustación en vivo mediante OBS / vMix).
* **Control interactivo**: Selector de tiros con actualización dinámica de porcentajes e indicador de quién tiene la ventaja.
* **Compatibilidad**: 100% cliente, sin dependencias de servidor ni frameworks pesados.

---

## 📖 Cita y Referencias Académicas

Si utilizas este trabajo o modelo en investigaciones relacionadas con analítica deportiva o tenis computacional:

`ibtex
@misc{soler2026barratenis,
  author       = {Federico Soler},
  title        = {Barra de Ventaja Táctica Dinámica para Tenis Profesional mediante XGBoost Secuencial},
  year         = {2026},
  publisher    = {Universidad de Palermo},
  howpublished = {https://github.com/federicosolerup/barra_ventaja_tenis}
}
`
