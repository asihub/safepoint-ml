"""
Test script for HuggingFace Inference API summarization (new router endpoint).
Run: python test_hf_summarization.py

Requires free HF token: https://huggingface.co/settings/tokens
  expo/rt HF_TOKEN=hf_your_token_here  (Linux/Mac)
  set HF_TOKEN=hf_your_token_here     (Windows)
"""

import urllib.request
import urllib.error
import json
import os
import time

MODELS = [
    ("Falconsai/text_summarization",   "~240MB fine-tuned T5"),
    ("sshleifer/distilbart-cnn-12-6",  "~900MB DistilBART"),
    ("facebook/bart-large-cnn",        "~1.6GB BART Large"),
]

TEST_TEXT = """
Mindfulness meditation is a mental training practice that teaches you to slow down racing thoughts,
let go of negativity, and calm both your mind and body. It combines meditation with the practice
of mindfulness, which can be defined as a mental state that involves being fully focused on the
present moment so you can acknowledge and accept your thoughts, feelings, and sensations without
judgment. Techniques can vary, but in general, mindfulness meditation involves deep breathing and
awareness of body and mind. Practicing mindfulness meditation does not require props or preparation.
You can practice anywhere, anytime. Research suggests that mindfulness meditation can reduce anxiety,
depression, and stress while improving overall mental health and wellbeing.
"""

HF_TOKEN = os.environ.get("HF_TOKEN", "HF_TOKEN_REMOVED")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set. Get a free token at https://huggingface.co/settings/tokens")
    print("Running anonymously — may hit rate limits.\n")

def test_model(model_id, description):
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"Size: {description}")
    print('='*60)

    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

    payload = json.dumps({
        "inputs": TEST_TEXT.strip(),
        "parameters": {"max_length": 80, "min_length": 30, "do_sample": False}
    }).encode()

    req = urllib.request.Request(url, data=payload, method='POST')
    req.add_header('Content-Type', 'application/json')
    if HF_TOKEN:
        req.add_header('Authorization', f'Bearer {HF_TOKEN}')

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed = time.time() - start
            result = json.loads(resp.read())

            if isinstance(result, list) and result:
                summary = result[0].get('summary_text', '')
                print(f"✓ Success in {elapsed:.1f}s")
                print(f"Summary: {summary}")
            elif isinstance(result, dict) and 'error' in result:
                print(f"✗ API Error: {result['error']}")
                if 'estimated_time' in result:
                    print(f"  Model loading, estimated {result['estimated_time']:.0f}s — retry in a moment")
            else:
                print(f"? Unexpected: {result}")

    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        body = e.read().decode()
        print(f"✗ HTTP {e.code} after {elapsed:.1f}s: {body[:300]}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error after {elapsed:.1f}s: {e}")

if __name__ == "__main__":
    print("HuggingFace Inference API Test (router endpoint)")
    print(f"Token: {'set ✓' if HF_TOKEN else 'not set'}")

    for model_id, description in MODELS:
        test_model(model_id, description)
        time.sleep(2)

    print("\nDone.")
