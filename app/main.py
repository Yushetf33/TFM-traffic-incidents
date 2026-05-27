from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.model import predict

app = FastAPI(
    title="Traffic Incident Classifier",
    description="API para clasificación automática de incidencias en tráfico urbano. "
                "Modelo: ResNet50 Fine-tuning. Clases: fluido, congestión, accidente, obras.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "mensaje": "Traffic Incident Classifier API",
        "version": "1.0.0",
        "uso": "POST /predict con una imagen para obtener la clasificación"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_incident(file: UploadFile = File(...)):
    # Validar que es una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (jpg, png, etc.)"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="No se pudo procesar la imagen. Verifica que el archivo sea válido."
        )
    
    result = predict(image)
    
    return JSONResponse(content={
        "archivo": file.filename,
        "prediccion": result["clase"],
        "confianza": result["confianza"],
        "probabilidades": result["probabilidades"]
    })