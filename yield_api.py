"""FastAPI server for yield_model.pkl

Run:  python -m uvicorn yield_api:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from google import genai
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("YIELD_MODEL_PATH", Path(__file__).parent / "yield_model.pkl"))
REQUIRED_SKLEARN = "1.6.1"      # the version the model was saved with
YIELD_UNIT = "t/ha"             # check against your training data
STATE = {}


def _force_cpu(model):
    """The model may have been trained on a GPU (e.g. Colab). Make it run on CPU."""
    try:
        reg = model.named_steps["xgb"].regressor_            # the fitted XGBRegressor
        reg.set_params(device="cpu")
        if str(reg.get_params().get("tree_method") or "").startswith("gpu"):
            reg.set_params(tree_method="hist")
        booster = reg.get_booster()
        booster.set_param({"device": "cpu"})
        if "gpu" in str(booster.attributes().get("tree_method", "")):
            booster.set_param({"tree_method": "hist"})
    except Exception as e:                                   # never block startup on this
        print("CPU override skipped:", e)


def load_model(path=MODEL_PATH):
    import sklearn
    if sklearn.__version__ != REQUIRED_SKLEARN:
        raise RuntimeError(
            f"yield_model.pkl needs scikit-learn=={REQUIRED_SKLEARN}, but this environment has "
            f"{sklearn.__version__} (Python {sys.version.split()[0]}). Use a Python 3.12 environment "
            f"and run: pip install -r requirements.txt")
    with open(path, "rb") as f:
        model = pickle.load(f)
    _force_cpu(model)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_model()
    pre = model.named_steps["pre"]
    cat_cols = list(pre.transformers_[0][2])                      # ['Crop', 'Season', 'State']
    cats = {c: [str(v) for v in vals]
            for c, vals in zip(cat_cols, pre.named_transformers_["cat"].categories_)}
    STATE.update(model=model, cats=cats, columns=list(model.feature_names_in_))
    yield
    STATE.clear()


app = FastAPI(title="AgroVision AI - Yield API", lifespan=lifespan)

# Open for development. Before deploying, replace "*" with your site's URL.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class YieldRequest(BaseModel):
    crop: str
    season: str
    state: str
    crop_year: int = Field(ge=1990, le=2100)
    area: float = Field(gt=0, description="hectares")
    annual_rainfall: float = Field(ge=0, description="mm")
    fertilizer: float = Field(ge=0, description="kg")
    pesticide: float = Field(ge=0, description="kg")
    lang: Optional[str] = None          # sent by the website, not used by the model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/api/yield/options")
def options():
    """Values the model knows. The website fills its dropdowns from this."""
    c = STATE["cats"]
    return {"crops": c["Crop"], "seasons": c["Season"], "states": c["State"]}


@app.post("/predict")
def predict(req: YieldRequest):
    cats = STATE["cats"]
    for value, col in ((req.crop, "Crop"), (req.season, "Season"), (req.state, "State")):
        if value not in cats[col]:
            raise HTTPException(422, f"Unknown {col.lower()} '{value}'. Allowed: {', '.join(cats[col])}")

    row = {
        "Crop": req.crop, "Crop_Year": req.crop_year, "Season": req.season, "State": req.state,
        "Area": req.area, "Annual_Rainfall": req.annual_rainfall,
        "Fertilizer": req.fertilizer, "Pesticide": req.pesticide,
    }
    X = pd.DataFrame([row])[STATE["columns"]]       # same column order as training
    y = float(STATE["model"].predict(X)[0])
    return {"predicted_yield": round(max(y, 0.0), 4), "unit": YIELD_UNIT, "inputs": row}

# ==========================================
# GEMINI AI FARMER ASSISTANT
# ==========================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    crop: str = "Unknown"
    location: str = "Unknown"


@app.post("/api/chat")
async def farmer_chat(request: ChatRequest):

    # Read the API key securely from the environment.
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured on the server."
        )

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
        You are AgriVision AI, a helpful agricultural assistant
        designed to support farmers.

        Farmer's crop: {request.crop}
        Farmer's location: {request.location}

        Farmer's question:
        {request.message}

        Instructions:
        1. Answer clearly and practically.
        2. Use simple language that farmers can understand.
        3. Give actionable agricultural guidance where appropriate.
        4. Do not invent weather forecasts, soil measurements,
           crop predictions, or disease diagnoses.
        5. If information is insufficient, ask for more details.
        6. For pesticide or fertilizer recommendations, emphasize
           following local regulations and product labels.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response."
            )

        return {
            "reply": answer,
            "crop": request.crop,
            "location": request.location
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Gemini API error: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=502,
            detail="The AI assistant could not generate a response."
        )