# Barra de Ventaja Táctica en Tenis Profesional 🎾

## 📄 Paper Científico
* 📥 **[Descargar / Leer Paper en PDF: Estimación de la probabilidad condicional de ganar un punto en tenis profesional](barra_ventaja_tenis_Soler_Federico_v9.pdf)**

## 📌 Descripción del Proyecto

A través del procesamiento del corpus del Match Charting Project (MCP) sobre partidos ATP masculinos entre jugadores diestros, se entrenó un modelo XGBoost que infiere tiro a tiro la probabilidad condicional de que el jugador al servicio gane el punto.

* 🔬 **[Colab](https://colab.research.google.com/github/federicosolerup/barra_ventaja_tenis/blob/main/pipeline.ipynb)**: Carga el modelo pre-entrenado y genera la simulación gráfica del caso de estudio.

---

## 📊 Caso de Estudio: Juan Martín del Potro vs. Roger Federer

Simulación paso a paso de la evolución de probabilidad de victoria en un punto bisagra (Final US Open 2009, 5to set, 4-2, 30-40, Break Point):

![Evolución de la Probabilidad Condicional de Victoria](overleaf/eval_bar_simulation.png)

---

## 🔬 Estudio de Ablación Temporal

| Modelo | Memoria Temporal | Variables | Accuracy | AUC-ROC | Brier Score | Log Loss | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Uniforme** | Ninguna (=0.50$) | 0 | 50.00% | 50.00% | 0.2500 | 0.6931 | 6.70% |
| **Dummy Mayoría** | Frecuentista (=0.567$) | 0 | 56.70% | 50.00% | 0.2455 | 0.6841 | 0.00% |
| **XGBoost (t)** | Impacto actual | 20 | 66.86% | 72.86% | 0.2106 | 0.6090 | 1.84% |
| **XGBoost (t, t-1)** | Impacto + Respuesta rival | 39 | 67.31% | 73.57% | 0.2078 | 0.6027 | 1.48% |
| **XGBoost (t, t-1, t-2)** | Impacto + Respuesta + Preparación | **58** | **67.65%** | **74.07%** | **0.2057** | **0.5982** | **1.21%** |

---

## 💻 Simulación Web Interactiva

El directorio /HTML contiene una aplicación web construida con tecnologías estándar (HTML5 Canvas, CSS Vanilla y JavaScript moderno) que permite introducir una secuencia de golpes manualmente y analizar como fluctúa la probabilidad de victoria de los jugadores tiro a tiro.

---

## 📖 Cita y Referencias Académicas

`ibtex
@misc{soler2026barratenis,
  author       = {Federico Soler},
  title        = {Estimación de la probabilidad condicional de ganar un punto en tenis profesional},
  year         = {2026},
  howpublished = {https://github.com/federicosolerup/barra_ventaja_tenis}
}
`
