# SafePoint ML Service

Python FastAPI service providing AI-powered mental health risk analysis and wellbeing resource summarization.

## Stack

- Python 3.11+
- FastAPI + Uvicorn
- DistilBERT (mental health risk classification)
- HuggingFace Inference API (wellbeing resource summarization)
- Trafilatura (web article extraction)

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Trafilatura (if not in requirements.txt yet)

```bash
pip install trafilatura
pip freeze > requirements.txt
```

### 5. Configure environment

Create `.env` file in `ml-service/` root:

```env
HF_TOKEN=hf_your_token_here
```

Get a free token at: https://huggingface.co/settings/tokens

### 6. Run the service

```bash
uvicorn src.main:app --reload --port 8001
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Mental health risk classification (DistilBERT) |
| POST | `/summarize` | Wellbeing resource summarization (HuggingFace BART) |

## Key dependencies

| Package | Purpose |
|---------|---------|
| `transformers` | DistilBERT model inference |
| `torch` | PyTorch backend |
| `fastapi` | REST API framework |
| `trafilatura` | Extract clean text from web articles |
| `requests` | HuggingFace Inference API calls |
| `python-dotenv` | Load `.env` variables |

## Summarization model

Uses `sshleifer/distilbart-cnn-12-6` via HuggingFace Inference API.
Model runs on HuggingFace servers — no local GPU required.

Summarization is triggered by Spring Boot scheduler (weekly) and stores results in PostgreSQL.

## Check installed versions

```bash
pip show trafilatura requests transformers torch fastapi
```

## Update requirements.txt after installing new packages

```bash
pip freeze > requirements.txt
```
