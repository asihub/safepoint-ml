import pandas as pd
import numpy as np

df = pd.read_csv(r"d:\Develop\General\safe-point\ml-service\data\dataset_clean.csv")

df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
df["token_est"]  = (df["word_count"] / 0.75).astype(int)

print("=== Довжина тексту по класах (слова) ===")
stats = df.groupby("Label")["word_count"].describe(percentiles=[0.5, 0.75, 0.95])
print(stats[["mean","min","max","50%","75%","95%"]].round(0))

print()
print("=== Розподіл по токенах ===")
bins   = [0, 128, 256, 512, 1024, 99999]
labels = ["<128", "128-256", "256-512", "512-1024", ">1024"]
df["token_bucket"] = pd.cut(df["token_est"], bins=bins, labels=labels)
print(df.groupby("Label")["token_bucket"].value_counts().sort_index())

print()
print("=== Скільки постів обрізається при MAX_LEN=256 ===")
cut = df[df["token_est"] > 256]
print(f"Всього: {len(cut)} з {len(df)} ({len(cut)/len(df)*100:.1f}%)")
print(cut.groupby("Label").size())