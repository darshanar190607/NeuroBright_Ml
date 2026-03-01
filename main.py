from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.predict_pipeline import PredictionPipeline
from src.exception import CustomException
import sys

app = FastAPI(title="Neuroadaptive EEG Emotion Detection")

templates = Jinja2Templates(directory="templates")

# Initialize prediction pipeline
try:
    pipeline = PredictionPipeline()
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    delta: float = Form(...),
    theta: float = Form(...),
    alpha: float = Form(...),
    beta: float = Form(...),
    gamma: float = Form(...)
):
    try:
        if pipeline is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Model not loaded. Please train the model first."
            })
        
        features = [delta, theta, alpha, beta, gamma]
        result = pipeline.predict_eeg_emotion(features)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "emotion": result["predicted_emotion"],
            "confidence": result["confidence_score"],
            "model_version": result["model_version"],
            "features": {
                "delta": delta,
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            }
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Prediction failed: {str(e)}"
        })

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
