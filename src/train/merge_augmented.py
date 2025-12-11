import pandas as pd

ORIG_PATH = r"d:\Develop\General\safe-point\ml-service\data\dataset_clean.csv"
AUG_PATH  = r"d:\Develop\General\safe-point\ml-service\data\dataset_augmentation_only.csv"
OUT_PATH  = r"d:\Develop\General\safe-point\ml-service\data\dataset_augmented.csv"

df_orig = pd.read_csv(ORIG_PATH)
df_aug  = pd.read_csv(AUG_PATH)

df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Оригінальний: {len(df_orig)}")
print(f"Augmented:    {len(df_aug)}")
print(f"Разом:        {len(df_combined)}")
print()
print("Розподіл класів:")
print(df_combined["label_id"].value_counts().sort_index().rename({0.0:"low",1.0:"medium",2.0:"high"}))

df_combined[["text","label_id","Label"]].to_csv(OUT_PATH, index=False)
print(f"\nЗбережено: {OUT_PATH}")