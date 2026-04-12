import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os

"""
04_train.py
───────────
Step 4 of 5 in the SafePoint ML training pipeline.

Fine-tunes distilbert-base-uncased on the merged SafePoint dataset
for 3-class mental health crisis risk classification (LOW / MEDIUM / HIGH).

Input:
    data/dataset_final.csv   — merged dataset from 03_merge_all_datasets.py
                               ~1,829 records | LOW: 786 | MEDIUM: 721 | HIGH: 222

Model:
    Base:         distilbert-base-uncased (~66M parameters)
    Task:         Sequence classification, 3 labels
    Output dir:   model/

Training configuration:
    MAX_LEN:           512 tokens (truncation applied to longer texts)
    BATCH_SIZE:        2   (safe for GTX 1650, 4.3 GB VRAM)
    GRAD_ACCUM_STEPS:  4   (effective batch size = 8)
    EPOCHS:            4   (with early stopping, patience=2)
    LEARNING_RATE:     2e-5
    SEED:              42

Class imbalance handling:
    sklearn compute_class_weight('balanced') applied to training set.
    HIGH class receives ~2.6x weight to compensate for underrepresentation.
    Loss function: CrossEntropyLoss with class weights.

Early stopping:
    Saves best model checkpoint when val_loss improves.
    Stops training if val_loss does not improve for 2 consecutive epochs.

What this script does:
    1. Loads and splits dataset (80% train / 20% val, stratified)
    2. Downloads distilbert-base-uncased tokenizer and model from HuggingFace
    3. Computes class weights and configures weighted loss
    4. Trains for up to 4 epochs with gradient accumulation
    5. Saves best checkpoint to model/ after each improvement
    6. Prints classification report and confusion matrix on val set

Output:
    model/   — fine-tuned DistilBERT model (overwrites previous version)
               make sure to back up model/ before running

Next step:
    Run 05_evaluate.py
"""

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH            = r"d:\Develop\General\safe-point\ml-service\data\dataset_final.csv"
MODEL_DIR            = r"d:\Develop\General\safe-point\ml-service\model"
MODEL_NAME           = "distilbert-base-uncased"

EPOCHS               = 4
BATCH_SIZE           = 2       # GTX 1650 — safe for MAX_LEN=512
GRAD_ACCUM_STEPS     = 4       # simulates batch_size=4 without additional VRAM
MAX_LEN              = 512
LR                   = 2e-5
SEED                 = 42

LABELS               = {0: "low", 1: "medium", 2: "high"}

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label_id"])
df["label_id"] = df["label_id"].astype(int)

print(f"\nClass distribution:")
print(df["label_id"].value_counts().sort_index().rename(LABELS))

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

# ── Tokenizer ───────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.tolist()
        self.labels    = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = MentalHealthDataset(train_df["text"], train_df["label_id"], tokenizer, MAX_LEN)
val_dataset   = MentalHealthDataset(val_df["text"],   val_df["label_id"],   tokenizer, MAX_LEN)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── Class weights (imbalance compensation) ───────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=train_df["label_id"].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy().round(2)}")

# ── Model ────────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3
)
model.to(device)

optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ── Training ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, accum_steps):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels) / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps
        preds       = outputs.logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            loss        = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds       = outputs.logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0
best_val_loss = float("inf")
patience = 2
no_improve = 0
print(f"\nMAX_LEN={MAX_LEN} | BATCH_SIZE={BATCH_SIZE} | GRAD_ACCUM={GRAD_ACCUM_STEPS} (effective batch={BATCH_SIZE*GRAD_ACCUM_STEPS})")
print("="*60)
print("Starting training")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    if device.type == "cuda":
        print(f"\nVRAM before epoch {epoch}: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, GRAD_ACCUM_STEPS)
    val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, loss_fn, device)

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"  Train — loss: {train_loss:.4f} | acc: {train_acc:.4f}")
    print(f"  Val   — loss: {val_loss:.4f}   | acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_acc = val_acc
        best_val_loss = val_loss
        no_improve = 0
        model.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  Early stopping — val_loss not improving for {patience} epochs")
            break

# ── Final report ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Classification Report (best model on val set)")
print("="*60)
print(classification_report(
    val_labels, val_preds,
    target_names=["low", "medium", "high"]
))

print("Confusion Matrix:")
print(confusion_matrix(val_labels, val_preds))
print(f"\nBest val accuracy: {best_val_acc:.4f}")
print(f"Model saved: {MODEL_DIR}")