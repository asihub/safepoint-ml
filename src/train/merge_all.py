import pandas as pd

DATA_DIR = r"d:\Develop\General\safe-point\ml-service\data"

# Всі датасети
files = {
    "C-SSRS original":  rf"{DATA_DIR}\dataset_clean.csv",
    "Augmented":        rf"{DATA_DIR}\dataset_augmentation_only.csv",
    "Reddit mapped":    rf"{DATA_DIR}\dataset_reddit_mapped.csv",
}

LABEL_NAME = {0: "low", 1: "medium", 2: "high"}

dfs = []
for name, path in files.items():
    try:
        df = pd.read_csv(path)
        print(f"{name}: {len(df)} записів")
        print(f"  {df['label_id'].value_counts().sort_index().rename(LABEL_NAME).to_dict()}")
        dfs.append(df)
    except FileNotFoundError:
        print(f"{name}: файл не знайдено — пропускаємо")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nФінальний датасет: {len(df_all)} записів")
print(df_all["label_id"].value_counts().sort_index().rename(LABEL_NAME))

out_path = rf"{DATA_DIR}\dataset_final.csv"
df_all[["text", "label_id", "Label"]].to_csv(out_path, index=False)
print(f"\nЗбережено: {out_path}")