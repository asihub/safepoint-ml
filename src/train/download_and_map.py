import pandas as pd
from datasets import load_dataset

OUT_DIR = r"d:\Develop\General\safe-point\ml-service\data"

# ── Завантаження ──────────────────────────────────────────────────────────────
print("Завантажуємо solomonk/reddit_mental_health_posts...")
ds = load_dataset("solomonk/reddit_mental_health_posts", split="train")
df = ds.to_pandas()

print(f"Загальна кількість: {len(df)}")
print(f"Subreddits: {df['subreddit'].value_counts().to_dict()}")

# ── Фільтрація ────────────────────────────────────────────────────────────────
df = df[df["body"].notna()]
df = df[~df["body"].isin(["[removed]", "[deleted]"])]

# ── Маппінг subreddit → label ─────────────────────────────────────────────────
label_map = {
    "ADHD":       0,
    "aspergers":  0,
    "OCD":        1,
    "ptsd":       1,
    "depression": 1,
}
label_name_map = {0: "Supportive", 1: "Ideation"}

df["label_id"] = df["subreddit"].map(label_map)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)
df["Label"] = df["label_id"].map(label_name_map)
df["text"] = df["body"].astype(str)

# ── Фільтр довжини ────────────────────────────────────────────────────────────
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df = df[(df["word_count"] >= 10) & (df["word_count"] <= 400)]

print(f"\nПісля фільтрації: {len(df)}")
print(df["label_id"].value_counts().sort_index().rename({0:"low", 1:"medium"}))

# ── Семплуємо рівну кількість на клас ────────────────────────────────────────
n_per_class = 500
parts = []
for label_id in [0, 1]:
    subset = df[df["label_id"] == label_id]
    sampled = subset.sample(min(len(subset), n_per_class), random_state=42)
    parts.append(sampled)

df_balanced = pd.concat(parts, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nПісля балансування: {len(df_balanced)}")
print(df_balanced["label_id"].value_counts().sort_index().rename({0:"low", 1:"medium"}))

# ── Зберігаємо ────────────────────────────────────────────────────────────────
out_path = rf"{OUT_DIR}\dataset_reddit_mapped.csv"
df_balanced[["text", "label_id", "Label"]].to_csv(out_path, index=False)
print(f"\nЗбережено: {out_path}")