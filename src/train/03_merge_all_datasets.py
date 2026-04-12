import pandas as pd

"""
03_merge_all_datasets.py
────────────────────────
Step 3 of 5 in the SafePoint ML training pipeline.

Merges all three data sources into a single shuffled dataset
ready for model training.

Input files:
    data/dataset_clean.csv              — C-SSRS dataset (Step 1)
                                          501 records, psychiatrist-annotated
                                          LOW: 108 | MEDIUM: 171 | HIGH: 122

    data/dataset_augmentation_only.csv  — Synthetic examples generated via Claude
                                          328 records, covering class imbalance gaps
                                          LOW: 178 | MEDIUM: 50  | HIGH: 100

    data/dataset_reddit_mapped.csv      — Reddit Mental Health Posts (Step 2)
                                          1,000 records, balanced 500/500
                                          LOW: 500 | MEDIUM: 500 | HIGH: —

What this script does:
    1. Loads all three source datasets
    2. Concatenates and shuffles (random_state=42 for reproducibility)
    3. Prints final class distribution
    4. Saves merged dataset → data/dataset_final.csv

Expected output:
    ~1,729 records
    LOW: ~786 | MEDIUM: ~721 | HIGH: ~222

Output:
    data/dataset_final.csv   — used by 04_train.py

Next step:
    Run 04_train.py
"""

DATA_DIR = r"../../data"

# All datasets
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
        print(f"{name}: {len(df)} records")
        print(f"  {df['label_id'].value_counts().sort_index().rename(LABEL_NAME).to_dict()}")
        dfs.append(df)
    except FileNotFoundError:
        print(f"{name}: file not found — skipping")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal dataset: {len(df_all)} records")
print(df_all["label_id"].value_counts().sort_index().rename(LABEL_NAME))

out_path = rf"{DATA_DIR}\dataset_final.csv"
df_all[["text", "label_id", "Label"]].to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")