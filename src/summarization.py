"""
summarization.py — Wellbeing resource summarization via HuggingFace Inference API.
Uses Trafilatura to extract clean text from URLs, then summarizes with BART.

Dependencies:
  pip install trafilatura requests
  HF_TOKEN env variable required.
"""

import os
import logging
import requests
import trafilatura
from dotenv import load_dotenv
import pathlib

# Load .env from ml-service root regardless of working directory
_ENV_PATH = pathlib.Path(__file__).parent.parent / '.env'
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_MODEL    = "sshleifer/distilbart-cnn-12-6"
HF_API_URL  = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
HF_HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

MAX_INPUT_CHARS  = 4000   # BART token limit ~1024, ~4000 chars is safe
SUMMARY_MAX_LEN  = 280    # longer to include practical tips
SUMMARY_MIN_LEN  = 100


def fetch_article_text(url: str) -> str | None:
    """Fetch and extract clean text from a URL using Trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning("Trafilatura: no content fetched from %s", url)
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text or len(text.strip()) < 100:
            logger.warning("Trafilatura: extracted text too short from %s", url)
            return None
        return text.strip()
    except Exception as e:
        logger.error("Trafilatura error for %s: %s", url, e)
        return None


def summarize_text(text: str) -> str | None:
    """Summarize text using HuggingFace BART via Inference API."""
    if not HF_TOKEN:
        logger.error("HF_TOKEN not set — cannot call HuggingFace API")
        return None

    # Truncate to avoid exceeding model token limit
    truncated = text[:MAX_INPUT_CHARS]

    try:
        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={
                "inputs": truncated,
                "parameters": {
                    "max_length": SUMMARY_MAX_LEN,
                    "min_length": SUMMARY_MIN_LEN,
                    "do_sample":  False,
                    "length_penalty": 2.0,
                    "num_beams": 4,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            return result[0].get("summary_text", "").strip()

        logger.error("Unexpected HF response: %s", result)
        return None

    except requests.HTTPError as e:
        logger.error("HuggingFace API HTTP error: %s — %s", e.response.status_code, e.response.text[:200])
        return None
    except Exception as e:
        logger.error("HuggingFace API error: %s", e)
        return None


def summarize_url(url: str) -> str | None:
    """Full pipeline: fetch URL → extract text → summarize."""
    logger.info("Summarizing: %s", url)

    text = fetch_article_text(url)
    if not text:
        return None

    summary = summarize_text(text)
    if summary:
        logger.info("Summary generated (%d chars) for %s", len(summary), url)
    return summary
