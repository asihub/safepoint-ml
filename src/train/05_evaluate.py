import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

"""
05_evaluate.py
──────────────
Step 5 of 5 in the SafePoint ML training pipeline.

Loads the fine-tuned DistilBERT model and runs inference on a set of
hand-crafted test messages covering all three risk classes and edge cases.

Input:
    model/   — fine-tuned DistilBERT model (output of 04_train.py)

Test cases (10 total):
    - Obvious LOW:       everyday stress, neutral check-ins
    - Borderline LOW:    mild distress, adjustment difficulties
    - Obvious MEDIUM:    passive ideation, hopelessness, no specific plan
    - Borderline MEDIUM: self-harm thoughts without action
    - Obvious HIGH:      active plan, suicide attempt, goodbye messages
    - Edge case:         proxy mode — caregiver describing someone else
                         (expected LOW — should not classify as HIGH)

What this script does:
    1. Loads tokenizer and model from model/
    2. Detects CUDA/CPU device
    3. Runs prediction on each test message
    4. Prints expected vs actual risk level with confidence and per-class scores
    5. Reports overall accuracy on the test set

Output:
    Console — prediction results table + accuracy score
    No files saved.
"""

MODEL_DIR = r"../../model"
MAX_LEN   = 256
LABELS    = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict(text):
    enc = tokenizer(text, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
    probs    = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    label_id = probs.argmax()
    return LABELS[label_id], round(float(probs[label_id]), 3), {
        "low": round(float(probs[0]), 3),
        "medium": round(float(probs[1]), 3),
        "high": round(float(probs[2]), 3)
    }

tests = [
    # Obvious LOW
    ("LOW",    "Just got back from vacation. Feeling rested and happy. Life is good right now."),
    ("LOW",    "Busy week at work but nothing unusual. Going to the gym tonight and cooking dinner."),
    # Borderline LOW/MEDIUM
    ("LOW",    "I've been feeling pretty stressed lately and not sleeping great. Nothing serious just a rough patch."),
    ("LOW",    "Feeling a bit lonely since I moved to a new city. Still getting settled. It takes time I know."),
    # Obvious MEDIUM
    ("MEDIUM", "I feel like nobody would notice if I disappeared. Everything feels pointless and I'm exhausted all the time."),
    ("MEDIUM", "I've been having thoughts about not wanting to be here anymore. Nothing specific just wishing I could escape everything."),
    # Borderline MEDIUM/HIGH
    ("MEDIUM", "I've been thinking about hurting myself but I haven't done anything. I'm scared of my own thoughts right now."),
    # Obvious HIGH
    ("HIGH",   "I've written a note and said goodbye to everyone. I've decided what I'm going to do and I'm at peace with it."),
    ("HIGH",   "I overdosed last night. I woke up and I'm angry that I'm still here. I'm thinking about trying again."),
    # Complex — supporting another person (should be LOW)
    ("LOW",    "My friend told me she's been having dark thoughts. I'm really worried about her and trying to be there for her."),
]

print("="*70)
print(f"{'Expected':<10} {'Result':<10} {'Confidence':<12} {'Scores'}")
print("="*70)
correct = 0
for expected, text in tests:
    risk, conf, scores = predict(text)
    match = "✓" if risk == expected else "✗"
    if risk == expected:
        correct += 1
    print(f"{match} {expected:<10} {risk:<10} {conf:<12} low={scores['low']} med={scores['medium']} high={scores['high']}")
    print(f"  {text[:80]}...")
    print()

print(f"Accuracy: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")