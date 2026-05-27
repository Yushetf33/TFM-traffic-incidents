# Clasificación de imágenes de tráfico urbano para detección automática de incidencias

TFM del Máster Deep Learning — Universidad Politécnica de Madrid

**Autor:** Yushetf López Jiménez  
**Directores:** Dr. Silvia Alba Uribe Mayoral, Gustavo Adolfo Hernández Peñaloza

## Descripción

Sistema de visión artificial basado en Deep Learning para la clasificación automática de escenas de tráfico urbano en 4 categorías:

- `fluido` — tráfico sin incidencias
- `congestion` — tráfico denso o detenido
- `accidente` — colisión o vehículo accidentado
- `obras` — trabajos en la vía

**Mejor modelo:** ResNet50 Fine-tuning — Accuracy 96.5% — Macro F1 0.968

## Estructura del repositorio

```
tfm-traffic-incidents/
  app/
    main.py       # API FastAPI
    model.py      # Carga del modelo y predicción
  models/         # Pesos del modelo entrenado (.pth)
  notebooks/      # Notebook de entrenamiento (Google Colab)
  requirements.txt
  README.md
```
## Datasets utilizados

| Dataset | Fuente | Clases aportadas |
|---|---|---|
| Cityscapes | cityscapes-dataset.com | fluido (perspectiva vehículo) |
| Traffic Density Singapore | Kaggle (rahat52) | fluido, congestión (cámara fija) |
| Car Crash | Roboflow (smith-fsuvw) | accidente |
| Road Construction | Roboflow (bus-hclqm) | obras |

## Modelos entrenados

| Modelo | Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN | 88.7% | 0.874 |
| EfficientNetB0 Transfer | 83.5% | 0.827 |
| ResNet50 Transfer | 87.0% | 0.868 |
| EfficientNetB0 Fine-tuning | 90.9% | 0.908 |
| **ResNet50 Fine-tuning** | **96.5%** | **0.968** |

## Instalación y uso

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abre [http://localhost:8000/docs](http://localhost:8000/docs) para la documentación interactiva.

### Ejemplo de uso con curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@imagen_trafico.jpg"
```

### Respuesta esperada

```json
{
  "archivo": "imagen_trafico.jpg",
  "prediccion": "congestion",
  "confianza": 0.9823,
  "probabilidades": {
    "accidente": 0.0021,
    "congestion": 0.9823,
    "fluido": 0.0134,
    "obras": 0.0022
  }
}
```

## Reproducibilidad

- Semilla fija: `SEED = 42`
- Entorno: Google Colab con GPU Tesla T4
- Framework: PyTorch 2.3.0 + torchvision 0.18.0
- El notebook completo está en la carpeta `notebooks/`
