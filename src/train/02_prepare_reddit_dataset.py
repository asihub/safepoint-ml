import pandas as pd
from datasets import load_dataset

"""
02_prepare_reddit_dataset.py
────────────────────────────
Step 2 of 6 in the SafePoint ML training pipeline.

Downloads and prepares the Reddit Mental Health Posts dataset from HuggingFace.
Provides LOW and MEDIUM risk examples to complement the C-SSRS dataset.

Source:
    HuggingFace: solomonk/reddit_mental_health_posts
    Total available: ~151,000 posts across 5 subreddits

Subreddit → risk class mapping:
    ADHD, aspergers → 0 (LOW)    — community support, non-crisis posts
    depression, ptsd, OCD → 1 (MEDIUM) — emotional distress, no acute crisis

What this script does:
    1. Downloads dataset from HuggingFace Hub
    2. Filters out removed/deleted posts and applies word count limits (10–400 words)
    3. Maps subreddits to numeric risk classes
    4. Balances classes by sampling 500 records per class
    5. Saves the balanced dataset → data/dataset_reddit_mapped.csv

Output:
    data/dataset_reddit_mapped.csv   — used by 04_merge_datasets.py

Next step:
    Run 03_augment_dataset.py
"""

OUT_DIR = r"../../data"

# ── Loading ──────────────────────────────────────────────────────────────
print("Loading solomonk/reddit_mental_health_posts...")
ds = load_dataset("solomonk/reddit_mental_health_posts", split="train")
df = ds.to_pandas()

print(f"Total count: {len(df)}")
print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")

# ── Filtering ────────────────────────────────────────────────────────────────
df = df[df["body"].notna()]
df = df[~df["body"].isin(["[removed]", "[deleted]"])]

# ── Mapping subreddit → label ─────────────────────────────────────────────────
label_map = {
    "ADHD": 0,
    "aspergers": 0,
    "OCD": 1,
    "ptsd": 1,
    "depression": 1,
}
label_name_map = {0: "Supportive", 1: "Ideation"}

df["label_id"] = df["subreddit"].map(label_map)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)
df["Label"] = df["label_id"].map(label_name_map)
df["text"] = df["body"].astype(str)

# ── Length filter ────────────────────────────────────────────────────────────
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df = df[(df["word_count"] >= 10) & (df["word_count"] <= 400)]

print(f"\nAfter filtering: {len(df)}")
print(df["label_id"].value_counts().sort_index().rename({0: "low", 1: "medium"}))

# ── Sample equal number per class ────────────────────────────────────────
n_per_class = 500
parts = []
for label_id in [0, 1]:
    subset = df[df["label_id"] == label_id]
    sampled = subset.sample(min(len(subset), n_per_class), random_state=42)
    parts.append(sampled)

df_balanced = pd.concat(parts, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAfter balancing: {len(df_balanced)}")
print(df_balanced["label_id"].value_counts().sort_index().rename({0: "low", 1: "medium"}))

# ── Saving ────────────────────────────────────────────────────────────────
out_path = rf"{OUT_DIR}\dataset_reddit_mapped.csv"
df_balanced[["text", "label_id", "Label"]].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
