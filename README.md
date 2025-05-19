# 🧠 Reconocimiento de Expresiones Faciales para Feedback en Clases

Este proyecto implementa un sistema inteligente que detecta expresiones faciales de estudiantes en tiempo real mediante una cámara. El objetivo es proporcionar **recomendaciones automáticas al docente** durante una clase, basadas en el estado emocional predominante del grupo.

---

## 📌 ¿Qué hace este programa?

- Captura video en tiempo real desde una cámara web.
- Detecta rostros y clasifica la emoción principal de cada uno.
- Muestra una **recomendación pedagógica inmediata** en pantalla.
- Al finalizar, muestra un **resumen de todas las emociones detectadas** y una sugerencia final basada en la emoción más frecuente.

---

## 🎯 Dataset utilizado

Se utilizó el dataset **FER2013** disponible en Kaggle:
🔗 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

Este dataset contiene imágenes en blanco y negro de rostros humanos clasificados en 7 emociones:  
`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

---

## 📁 Estructura del proyecto

```
facial_emotion_recognition/
│
├── data/
│   ├── train/            # Imágenes de entrenamiento organizadas por clase
│   └── test/             # Imágenes de prueba organizadas por clase
│
├── models/
│   └── modelo_entrenado.h5  # Modelo entrenado con TensorFlow (CNN)
│
├── src/
│   ├── preprocesamiento.py        # Utilidades para leer/preparar imágenes individuales
│   ├── entrenamiento.py           # Entrena el modelo desde cero (CNN)
│   ├── evaluacion.py              # Evalúa el modelo con métricas (precision, recall, F1)
│   ├── interfaz_tkinter.py        # Interfaz gráfica en Tkinter con cámara y recomendaciones
│
├── main.py                  # Ventana principal para usar el sistema completo
├── requirements.txt         # Dependencias del sistema
├── README.md                # Este documento
```

---

## 🧩 Explicación de cada archivo

| Archivo                     | Descripción |
|----------------------------|-------------|
| `main.py`                  | Lanza una ventana gráfica con botones para entrenar, evaluar o iniciar la cámara. |
| `src/entrenamiento.py`     | Entrena un modelo CNN desde cero usando el dataset de imágenes. |
| `src/evaluacion.py`        | Evalúa el modelo con métricas como F1-score y muestra la matriz de confusión. |
| `src/interfaz_tkinter.py`  | Abre una ventana con cámara en vivo que muestra emociones detectadas y da recomendaciones. |
| `src/preprocesamiento.py`  | Función auxiliar para cargar y preparar imágenes para inferencia. |
| `models/`                  | Carpeta donde se guarda el modelo entrenado (`modelo_entrenado.h5`). |
| `data/`                    | Contiene el dataset organizado por carpetas de emociones. |
| `requirements.txt`         | Lista de paquetes necesarios para ejecutar el sistema. |

---

## 🚀 Cómo ejecutar el sistema

### 1. Instala las dependencias

Verifica la versión de python que tienes instalado, ya que esta versión fue creada con python 3.10.0. Lo cual hay algunas librerias que no tienen aún soporte para las ultimas versiones de python. Puedes verificarlo con el siguiente comando:

```bash
python --version
```

Y esto debera arrojar algo como `Python versión 3.10.0`

Posteriormente de revisar la versión ya podemos instalar el archivo `requirements.txt`

```bash
pip install -r requirements.txt
```

### 2. Ejecuta la interfaz gráfica principal
```bash
python main.py
```

Desde allí podrás:
- Entrenar el modelo (CNN)
- Evaluar el modelo con métricas
- Iniciar la cámara con detección y recomendaciones

---

## ✅ Requisitos técnicos

- Python 3.7+
- TensorFlow / Keras
- OpenCV
- Tkinter
- NumPy
- Pillow
- scikit-learn

---

## 📊 Resultado esperado

- Se muestran emociones detectadas en tiempo real.
- Se genera una recomendación automática en la interfaz.
- Al presionar "Finalizar", se presenta un **resumen de emociones** y una **recomendación final al docente**.
