# safepoint-ml

> Python FastAPI ML service for SafePoint — provides real-time mental health crisis risk classification using a fine-tuned DistilBERT model, and AI-generated summaries of wellbeing resources using BART via the HuggingFace Inference API.

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| API Framework | FastAPI + Uvicorn |
| Risk Classification | DistilBERT (distilbert-base-uncased), fine-tuned |
| Summarization | sshleifer/distilbart-cnn-12-6 via HuggingFace Inference API |
| Text Extraction | Trafilatura |
| ML Framework | PyTorch 2.7 + HuggingFace Transformers 5.x |

---

## Model

### Risk Classification (DistilBERT)

A custom fine-tuned sequence classification model trained on 1,729 labeled examples from three sources:

| Dataset | Records | Source |
|---|---|---|
| C-SSRS annotated Reddit posts | 500 | Kaggle — labeled by practicing psychiatrists |
| Reddit mental health posts | 1,000 | Kaggle — community posts mapped to risk levels |
| LLM-augmented synthetic examples | 229 | Generated to address class imbalance |

**Training configuration:**
- Base model: `distilbert-base-uncased` (~66M parameters)
- 3-class output: LOW · MEDIUM · HIGH
- Class-weighted CrossEntropyLoss (compensates for HIGH class being 3× smaller)
- Gradient accumulation (4 steps) to fit within 4.3 GB VRAM
- Early stopping (patience = 2)
- Hardware: NVIDIA GTX 1650, CUDA 11.8

**Evaluation results:**

| Metric | Value |
|---|---|
| Validation accuracy | 0.82 |
| Macro F1 | 0.80 |
| F1 — LOW | 0.88 |
| F1 — MEDIUM | 0.80 |
| F1 — HIGH | 0.72 |
| Manual test accuracy | 9/10 (90%) |

### Summarization (BART)

Uses `sshleifer/distilbart-cnn-12-6` via HuggingFace Inference API.  
Article text is extracted with Trafilatura, truncated to 4,000 characters, and summarized to 100–280 tokens.  
Runs on HuggingFace servers — no local GPU required for summarization.

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/analyze` | Mental health risk classification (DistilBERT) |
| POST | `/summarize` | Wellbeing resource summarization (BART via HuggingFace API) |
| GET | `/health` | Service status and model load state |

### POST /analyze

Request:
```json
{ "text": "I feel completely hopeless and I don't see the point anymore." }
```

Response:
```json
{
  "risk_level": "MEDIUM",
  "confidence": 0.913,
  "scores": { "low": 0.022, "medium": 0.913, "high": 0.065 },
  "signals": ["hopelessness"]
}
```

Detected signal categories: `hopelessness` · `isolation` · `self_harm` · `suicidal_ideation` · `plan_or_action` · `burden_feeling`

### POST /summarize

Request:
```json
{ "url": "https://www.nami.org/About-Mental-Illness/Treatments/..." }
```

Response:
```json
{ "url": "...", "excerpt": "AI-generated summary of the article..." }
```

### GET /health
```json
{ "status": "ok", "model_loaded": true, "device": "cuda" }
```

---

## Integration Notes

The ML service is called by the Java Spring Boot API (`safepoint-api`), not directly by the frontend.

- **Bound to** `127.0.0.1:8001` — not accessible from the public internet
- **Text quality gate** applied by Java API before calling this service: ≥ 15 words and ≥ 8 unique words required
- **Confidence threshold** applied by Java API after response: ML signal ignored if confidence < 0.60
- **Summarization** triggered weekly by Spring `@Scheduled` job (Sunday 2am); results stored in PostgreSQL

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create `.env` in the `ml-service/` root (never commit this file):

```env
HF_TOKEN=hf_your_token_here
MODEL_DIR=/path/to/model
```

Get a free HuggingFace token at: https://huggingface.co/settings/tokens

### 4. Run the service

```bash
python src/main.py
```

Service starts on `http://127.0.0.1:8001`  
Interactive docs: `http://127.0.0.1:8001/docs`

---

## Related Repositories

| Repository | Description |
|---|---|
| [safepoint-api](https://github.com/asihub/safepoint-api) | Java Spring Boot API (orchestrator) |
| [safepoint-ui](https://github.com/asihub/safepoint-ui) | React 19 frontend |
