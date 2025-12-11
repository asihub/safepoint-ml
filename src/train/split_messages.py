import pandas as pd
import numpy as np
import ast

RAW_PATH   = r"d:\Develop\General\safe-point\ml-service\data\500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv"
OUT_PATH   = r"d:\Develop\General\safe-point\ml-service\data\dataset_messages.csv"

label_map = {
    "Supportive": 0,
    "Ideation":   1,
    "Behavior":   2,
    "Attempt":    2,
}

# ── Завантаження ──────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print(f"Оригінальних записів: {len(df)}")

# ── Розбивка на окремі повідомлення ──────────────────────────────────────────
rows = []
for _, row in df.iterrows():
    label    = row["Label"]
    label_id = label_map.get(label)
    if label_id is None:
        continue
    try:
        messages = ast.literal_eval(row["Post"])
    except Exception:
        messages = [str(row["Post"])]

    for msg in messages:
        text = str(msg).strip()
        if len(text.split()) >= 5:  # фільтр дуже коротких
            rows.append({
                "text":     text,
                "label_id": label_id,
                "Label":    label
            })

result = pd.DataFrame(rows)

# ── Статистика ────────────────────────────────────────────────────────────────
print(f"\nПісля розбивки: {len(result)} повідомлень")
print(f"\nРозподіл класів:")
print(result["label_id"].value_counts().sort_index().rename({0:"low",1:"medium",2:"high"}))

result["word_count"]  = result["text"].apply(lambda x: len(x.split()))
result["token_est"]   = (result["word_count"] / 0.75).astype(int)

print(f"\nДовжина повідомлень (слова):")
print(result.groupby("Label")["word_count"].describe(
    percentiles=[0.5, 0.75, 0.95]
)[["mean","min","max","50%","75%","95%"]].round(0))

print(f"\nРозподіл по токенах:")
bins   = [0, 128, 256, 512, 99999]
labels = ["<128", "128-256", "256-512", ">512"]
result["token_bucket"] = pd.cut(result["token_est"], bins=bins, labels=labels)
print(result["token_bucket"].value_counts().sort_index())

print(f"\nОбрізається при MAX_LEN=512: {len(result[result['token_est']>512])} ({len(result[result['token_est']>512])/len(result)*100:.1f}%)")

# ── Збереження ────────────────────────────────────────────────────────────────
result[["text", "label_id", "Label"]].to_csv(OUT_PATH, index=False)
print(f"\nЗбережено: {OUT_PATH}")