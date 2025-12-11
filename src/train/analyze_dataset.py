import pandas as pd
import ast
import matplotlib.pyplot as plt

DATA_DIR = r"d:\Develop\General\safe-point\ml-service\data"
CSV_PATH = rf"{DATA_DIR}\500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv"

# -- 1. Завантаження
df = pd.read_csv(CSV_PATH)
print(f"Загальна кількість записів: {len(df)}")
print(f"Колонки: {df.columns.tolist()}")
print()

# -- 2. Розподіл класів
print("=== Розподіл класів ===")
label_counts = df["Label"].value_counts()
print(label_counts)
print()

# -- 3. Парсинг Post (масив рядків -> один текст)
def parse_post(raw):
    try:
        messages = ast.literal_eval(raw)
        return " ".join(messages) if isinstance(messages, list) else str(raw)
    except Exception:
        return str(raw)

df["text"] = df["Post"].apply(parse_post)

# -- 4. Аналіз довжини тексту
df["char_len"] = df["text"].apply(len)
df["word_len"] = df["text"].apply(lambda x: len(x.split()))

print("=== Довжина тексту (символи) ===")
print(df.groupby("Label")["char_len"].describe()[["mean", "min", "max", "50%"]])
print()

print("=== Довжина тексту (слова) ===")
print(df.groupby("Label")["word_len"].describe()[["mean", "min", "max", "50%"]])
print()

# -- 5. Порогове значення 512 токенів (~400 слів)
# DistilBERT: 1 токен ~ 0.75 слова
df["estimated_tokens"] = (df["word_len"] / 0.75).astype(int)
over_limit = df[df["estimated_tokens"] > 512]
print(f"=== Записи що перевищують 512 токенів: {len(over_limit)} ({len(over_limit)/len(df)*100:.1f}%) ===")
print(over_limit["Label"].value_counts())
print()

# -- 6. Пусті записи
print("=== Пусті записи ===")
print(df["text"].isnull().sum(), "null")
print((df["text"] == "").sum(), "порожніх рядків")
print()

# -- 7. Маппінг міток
label_map = {
    "Supportive": 0,  # low
    "Ideation":   1,  # medium
    "Behavior":   2,  # high
    "Attempt":    2,  # high
}
df["label_id"] = df["Label"].map(label_map)

print("=== Після маппінгу (0=low, 1=medium, 2=high) ===")
print(df["label_id"].value_counts())
print()

# -- 8. Графік розподілу класів
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#FFC107", "#F44336", "#9C27B0"])
axes[0].set_title("Оригінальні мітки")
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Кількість")
axes[0].tick_params(axis='x', rotation=0)

df["label_id"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color=["#4CAF50", "#FFC107", "#F44336"]
)
axes[1].set_title("Після маппінгу (0=low, 1=medium, 2=high)")
axes[1].set_xlabel("Label ID")
axes[1].set_ylabel("Кількість")
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(rf"{DATA_DIR}\label_distribution.png", dpi=150)
print(f"Графік збережено: {DATA_DIR}\\label_distribution.png")

# -- 9. Зберегти очищений датасет
df[["text", "label_id", "Label"]].to_csv(rf"{DATA_DIR}\dataset_clean.csv", index=False)
print(f"Очищений датасет збережено: {DATA_DIR}\\dataset_clean.csv")