import ast

import matplotlib.pyplot as plt
import pandas as pd

"""
01_prepare_cssrs_dataset.py
────────────────────────
Step 1 of 4 in the SafePoint ML training pipeline.

Loads and prepares the C-SSRS (Columbia Suicide Severity Rating Scale) dataset —
500 anonymized Reddit posts labeled by practicing psychiatrists.

Source:
    Kaggle: thedevastator/c-ssrs-labeled-suicidality-in-500-anonymized-reddit-posts
    File:   500_anonymized_Reddit_users_posts_labels.csv

What this script does:
    1. Loads the raw CSV and prints class distribution
    2. Parses the Post column (stored as a Python list string) into plain text
    3. Analyzes text length (words and estimated DistilBERT tokens)
    4. Reports how many records exceed the 512-token model limit
    5. Maps original labels to numeric risk classes:
           Supportive → 0 (LOW)
           Ideation   → 1 (MEDIUM)
           Behavior   → 2 (HIGH)
           Attempt    → 2 (HIGH)
    6. Saves a class distribution chart → data/label_distribution.png
    7. Saves the cleaned dataset        → data/dataset_clean.csv

Output:
    data/dataset_clean.csv   — used by 03_merge_all_datasets.py (Step 4)

Next step:
    Run 02_prepare_reddit_dataset.py (Step 2)
"""

DATA_DIR = r"d:\Develop\General\safe-point\ml-service\data"
CSV_PATH = rf"{DATA_DIR}\500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv"

# -- 1. Loading
df = pd.read_csv(CSV_PATH)
print(f"Total number of records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print()

# -- 2. Class distribution
print("=== Class Distribution ===")
label_counts = df["Label"].value_counts()
print(label_counts)
print()


# -- 3. Parse Post (array of strings -> single text)
def parse_post(raw):
    try:
        messages = ast.literal_eval(raw)
        return " ".join(messages) if isinstance(messages, list) else str(raw)
    except Exception:
        return str(raw)


df["text"] = df["Post"].apply(parse_post)

# -- 4. Text length analysis
df["char_len"] = df["text"].apply(len)
df["word_len"] = df["text"].apply(lambda x: len(x.split()))

print("=== Text Length (characters) ===")
print(df.groupby("Label")["char_len"].describe()[["mean", "min", "max", "50%"]])
print()

print("=== Text Length (words) ===")
print(df.groupby("Label")["word_len"].describe()[["mean", "min", "max", "50%"]])
print()

# -- 5. Threshold of 512 tokens (~400 words)
# DistilBERT: 1 token ~ 0.75 words
df["estimated_tokens"] = (df["word_len"] / 0.75).astype(int)
over_limit = df[df["estimated_tokens"] > 512]
print(f"=== Records exceeding 512 tokens: {len(over_limit)} ({len(over_limit) / len(df) * 100:.1f}%) ===")
print(over_limit["Label"].value_counts())
print()

# -- 6. Empty records
print("=== Empty Records ===")
print(df["text"].isnull().sum(), "null")
print((df["text"] == "").sum(), "empty strings")
print()

# -- 7. Label mapping
label_map = {
    "Supportive": 0,  # low
    "Ideation": 1,  # medium
    "Behavior": 2,  # high
    "Attempt": 2,  # high
}
df["label_id"] = df["Label"].map(label_map)

print("=== After mapping (0=low, 1=medium, 2=high) ===")
print(df["label_id"].value_counts())
print()

# -- 8. Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#FFC107", "#F44336", "#9C27B0"])
axes[0].set_title("Original Labels")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=0)

df["label_id"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color=["#4CAF50", "#FFC107", "#F44336"]
)
axes[1].set_title("After Mapping (0=low, 1=medium, 2=high)")
axes[1].set_xlabel("Label ID")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(rf"{DATA_DIR}\label_distribution.png", dpi=150)
print(f"Plot saved: {DATA_DIR}\\label_distribution.png")

# -- 9. Save cleaned dataset
df[["text", "label_id", "Label"]].to_csv(rf"{DATA_DIR}\dataset_clean.csv", index=False)
print(f"Cleaned dataset saved: {DATA_DIR}\\dataset_clean.csv")
