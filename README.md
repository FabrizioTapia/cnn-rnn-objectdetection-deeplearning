# Trabajo Práctico N°2 - Deep Learning Aplicado
## Redes Convolucionales, Detección de Objetos y Redes Recurrentes

**Tecnicatura Universitaria en Inteligencia Artificial - FCEIA - UNR**

Este trabajo práctico implementa y optimiza modelos de Deep Learning para clasificación de imágenes, detección de objetos y reconocimiento de comandos de voz (CRNN).

---

## 📋 Descripción General

El TP2 consiste en tres ejercicios principales que cubren diferentes arquitecturas de redes neuronales:

1. **Ejercicio 1**: Clasificación de razas de perros con CNN desde cero
2. **Ejercicio 2**: Detección de señales de tránsito con YOLO y Transfer Learning
3. **Ejercicio 3**: Reconocimiento de comandos de voz con RNN/CRNN

---

## 🛠️ Requisitos y Configuración

### Dependencias Principales
```python
# Core Deep Learning
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0

# Computer Vision
opencv-python>=4.5.0
Pillow>=8.0.0

# Object Detection
ultralytics>=8.0.0

# Data Processing
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Audio Processing (Ejercicio 3)
librosa>=0.8.0
scipy>=1.7.0
```

### Configuración de Semillas (Reproducibilidad)
```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

## 📁 Estructura del Proyecto

```
Completar #######################################################################################################
```

---

## 🐕 Ejercicio 1: Clasificación de Razas de Perros

### Dataset
- **Nombre**: 70 Dog Breeds Image Dataset
- **Clases**: 70 razas de perros (recomendado reducir a 5-10 para experimentación)

### Experimentos Requeridos

| Experimento | Descripción | Modificaciones |
|-------------|-------------|----------------|
| **E1** | Línea base | CNN simple: 3 bloques Conv-BN-ReLU-MaxPool |
| **E2** | Arquitectura profunda | Más capas convolucionales, filtros 3×3 vs 5×5 |
| **E3** | Bloques avanzados | Skip connections (ResNet) o módulos Inception |
| **E4** | Regularización | Dropout, BatchNorm, Data Augmentation |

### Preparación de Datos
- Resize a resolución fija (128×128 o 224×224)
- Normalización de píxeles
- Data Augmentation solo en conjunto de entrenamiento

### Métricas a Reportar
Para cada experimento:
- Accuracy en Train/Validation/Test
- Número de parámetros
- Épocas de entrenamiento
- Observaciones y análisis

Para el mejor modelo:
- Curvas de loss y accuracy
- Matriz de confusión
- Ejemplos de predicciones correctas/incorrectas
- Análisis de razas confundidas

---

## 🚦 Ejercicio 2: Detección de Señales de Tránsito

### Dataset
- **Nombre**: Traffic Signs Detection
- **Clases (15)**: Green Light, Red Light, Speed Limit (10-120), Stop
- **Formato**: YOLO Ultralytics

### Estructura del Dataset
```
dataset/
├── data.yaml          # Configuración: rutas, clases
├── train/
│   ├── images/        # Imágenes de entrenamiento
│   └── labels/        # Anotaciones .txt (YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Tareas
1. **Inspección del dataset**: Entender estructura y distribución
2. **Configuración YOLO**: Adaptar al dataset específico
3. **Transfer Learning**: Fine-tuning de modelo pre-entrenado
4. **Evaluación**: Métricas de detección (mAP, precision, recall)

---

## 🎤 Ejercicio 3: Reconocimiento de Comandos de Voz

### Dataset
- **Nombre**: Speech Commands Dataset
- **Comandos**: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
- **Formato**: Archivos de audio .wav

### Arquitecturas a Explorar
- **RNN Simple**: Vanilla RNN para secuencias de audio
- **LSTM/GRU**: Redes recurrentes avanzadas
- **CRNN**: Convolucional + Recurrente para espectrogramas

### Procesamiento de Audio
- Extracción de características (MFCC, espectrogramas)
- Normalización de señales
- Data Augmentation (ruido, desplazamiento temporal)

---

## 📊 Buenas Prácticas y Metodología

### Checkpoints y Reanudación
```python
# Guardar checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f'checkpoint_epoch_{epoch}.pth')

# Retomar desde checkpoint
checkpoint = torch.load('checkpoint_epoch_N.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### División de Datos
- **Train**: Único conjunto visible durante entrenamiento
- **Validation**: Para monitoreo y decisiones de hiperparámetros
- **Test**: Uso único al final para métricas definitivas

### Documentación
Cada ejercicio debe entregarse como notebook Jupyter con:
- Celdas de código implementadas
- Celdas Markdown explicando decisiones
- Análisis de resultados y experimentos
- Visualizaciones y métricas

---

## 🚀 Ejecución del Proyecto

### Instalación
```bash
# Clonar repositorio
git clone <repository-url>
cd cnn-rnn-objectdetection-deeplearning

# Crear entorno virtual
´´´######################################################################################################################


# Instalar dependencias
pip install -r requirements.txt
```

---

## 📈 Evaluación y Resultados

### Métricas Principales
- **Clasificación**: Accuracy, Precision, Recall, F1-Score
- **Detección**: mAP@0.5, mAP@0.5:0.95, precision, recall
- **Reconocimiento de Voz**: Accuracy por comando, matriz de confusión

### Análisis Cualitativo
- Visualización de activaciones
- Análisis de errores comunes
- Comparación entre arquitecturas
- Impacto de técnicas de regularización

---


## 📚 Referencias y Recursos


---

**Autores**: Fabrizio Tapia, Caterina Martinez Dufour, Damián Grimaldi
**Curso**: Machine Learning 2 - TUIA  
**Año**: 2026
