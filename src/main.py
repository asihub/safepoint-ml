from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging
import os

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safepoint-ml")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", r"d:\Develop\General\safe-point\ml-service\model")
MAX_LEN   = int(os.getenv("MAX_LEN", "512"))
HOST      = os.getenv("HOST", "127.0.0.1")  # internal only
PORT      = int(os.getenv("PORT", "8001"))

LABELS    = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# ── Signals для explainability ────────────────────────────────────────────────
SIGNAL_PATTERNS = {
    "hopelessness":   ["hopeless", "no hope", "pointless", "no point", "nothing matters"],
    "isolation":      ["alone", "nobody cares", "no one", "isolated", "invisible"],
    "self_harm":      ["hurt myself", "cutting", "self-harm", "harming myself"],
    "suicidal_ideation": ["end it", "not be here", "don't want to live", "wish i was dead",
                          "want to die", "kill myself", "suicidal"],
    "plan_or_action": ["bought", "pills", "rope", "gun", "method", "note", "goodbye",
                       "decided", "plan", "attempt", "overdose"],
    "burden_feeling": ["burden", "better off without me", "everyone would be better"],
}

def detect_signals(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for signal, keywords in SIGNAL_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(signal)
    return found

# ── Model state ───────────────────────────────────────────────────────────────
class ModelState:
    tokenizer: DistilBertTokenizer = None
    model: DistilBertForSequenceClassification = None
    device: torch.device = None

state = ModelState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Loading model from {MODEL_DIR}...")
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {state.device}")
    state.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    state.model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    state.model.to(state.device)
    state.model.eval()
    logger.info("Model loaded and ready")
    yield
    # Shutdown
    logger.info("Shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SafePoint ML Service",
    description="Mental health crisis risk classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # disable in prod: docs_url=None
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Java API only
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000,
                      description="Free-text input from user")

class Scores(BaseModel):
    low:    float
    medium: float
    high:   float

class AnalyzeResponse(BaseModel):
    risk_level:  str          # LOW / MEDIUM / HIGH
    confidence:  float
    scores:      Scores
    signals:     list[str]    # detected crisis signals

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "device": str(state.device)
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        enc = state.tokenizer(
            req.text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = state.model(
                input_ids=enc["input_ids"].to(state.device),
                attention_mask=enc["attention_mask"].to(state.device)
            ).logits

        probs    = F.softmax(logits, dim=1).squeeze().cpu().tolist()
        label_id = int(torch.argmax(torch.tensor(probs)).item())
        signals  = detect_signals(req.text)

        # Не логуємо текст — тільки метрики
        logger.info(f"Analyzed: risk={LABELS[label_id]} conf={probs[label_id]:.3f} signals={signals}")

        return AnalyzeResponse(
            risk_level=LABELS[label_id],
            confidence=round(probs[label_id], 3),
            scores=Scores(
                low=round(probs[0], 3),
                medium=round(probs[1], 3),
                high=round(probs[2], 3),
            ),
            signals=signals
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)