import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

MODEL_DIR = r"d:\Develop\General\safe-point\ml-service\model"
MAX_LEN   = 256
LABELS    = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict(text: str):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device)
        ).logits
    probs      = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    label_id   = probs.argmax()
    return {
        "risk_level": LABELS[label_id],
        "confidence": round(float(probs[label_id]), 3),
        "scores": {
            "low":    round(float(probs[0]), 3),
            "medium": round(float(probs[1]), 3),
            "high":   round(float(probs[2]), 3),
        }
    }

# ── Тестові приклади ──────────────────────────────────────────────────────────
examples = [
    "I've been feeling a bit tired lately but overall things are okay.",
    "I can't stop crying. Everything feels hopeless and I don't see the point anymore.",
    "I've been having dark thoughts. I bought pills last week and I keep thinking about using them.",
    "Work has been stressful but I'm managing. I just need a break.",
    "I tried to hurt myself last night. I don't want to be here anymore.",
]

print("="*60)
print("SafePoint — Risk Prediction Test")
print("="*60)
for text in examples:
    result = predict(text)
    print(f"\nText: {text[:80]}...")
    print(f"Risk:       {result['risk_level']} (confidence: {result['confidence']})")
    print(f"Scores:     low={result['scores']['low']} | medium={result['scores']['medium']} | high={result['scores']['high']}")

print("\n" + "="*60)
print("Введи свій текст (або 'exit' для виходу):")
while True:
    text = input("\n> ").strip()
    if text.lower() == "exit":
        break
    if text:
        result = predict(text)
        print(f"Risk:   {result['risk_level']} (confidence: {result['confidence']})")
        print(f"Scores: low={result['scores']['low']} | medium={result['scores']['medium']} | high={result['scores']['high']}")