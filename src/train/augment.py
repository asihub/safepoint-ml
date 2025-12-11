import anthropic
import pandas as pd
import json
import time
import os

# ── Конфігурація ─────────────────────────────────────────────────────────────
DATA_PATH  = r"d:\Develop\General\safe-point\ml-service\data\dataset_clean.csv"
OUT_PATH   = r"d:\Develop\General\safe-point\ml-service\data\dataset_augmented.csv"
API_KEY    = "ANTHROPIC_KEY_REMOVED"

# Скільки нових прикладів генерувати на клас
SAMPLES_PER_CLASS = 100

LABEL_MAP = {0: "low", 1: "medium", 2: "high"}

# ── Промпти для кожного класу ─────────────────────────────────────────────────
PROMPTS = {
    0: """Generate {n} short Reddit-style messages (50-150 words each) written by someone providing emotional support to another person in distress. 
The tone should be warm, encouraging, and empathetic — offering hope, practical advice, or simply listening.
Do NOT include suicidal ideation or crisis signals.

Return ONLY a JSON array of strings, no other text:
["message 1", "message 2", ...]""",

    1: """Generate {n} short Reddit-style messages (50-150 words each) written by someone experiencing moderate emotional distress, sadness, or passive suicidal ideation.
Examples of signals: feeling hopeless, worthless, lonely, tired of life, wishing things were different, vague thoughts of not wanting to be here.
No specific plans or methods. No past attempts.

Return ONLY a JSON array of strings, no other text:
["message 1", "message 2", ...]""",

    2: """Generate {n} short Reddit-style messages (50-150 words each) written by someone in serious mental health crisis.
Examples of signals: active suicidal ideation with some intent, specific plans or methods mentioned, past suicide attempts, self-harm behaviors, statements like "I've decided", "I can't go on", "I've written a note".
These should represent high-risk situations requiring immediate intervention.

Return ONLY a JSON array of strings, no other text:
["message 1", "message 2", ...]"""
}

def generate_samples(client, label_id, n):
    prompt = PROMPTS[label_id].format(n=n)
    print(f"  Генеруємо {n} прикладів для класу '{LABEL_MAP[label_id]}'...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()

    # Очищаємо якщо є markdown backticks
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    messages = json.loads(text)
    return messages

def main():
    client = anthropic.Anthropic(api_key=API_KEY)

    # Завантажуємо оригінальний датасет
    df_orig = pd.read_csv(DATA_PATH)
    print(f"Оригінальний датасет: {len(df_orig)} записів")
    print(df_orig["label_id"].value_counts().sort_index().rename(LABEL_MAP))

    new_rows = []

    for label_id in [0, 1, 2]:
        try:
            messages = generate_samples(client, label_id, SAMPLES_PER_CLASS)
            for msg in messages:
                # Перевірка довжини — фільтруємо якщо більше ~400 слів
                word_count = len(str(msg).split())
                if 10 <= word_count <= 400:
                    new_rows.append({
                        "text":     str(msg).strip(),
                        "label_id": label_id,
                        "Label":    ["Supportive", "Ideation", "Behavior"][label_id]
                    })
            print(f"  ✓ Отримано {len(messages)} прикладів")
            time.sleep(1)  # rate limit
        except Exception as e:
            print(f"  ✗ Помилка для класу {label_id}: {e}")

    df_new = pd.DataFrame(new_rows)
    print(f"\nНових прикладів: {len(df_new)}")

    # Об'єднуємо з оригінальним датасетом
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nКомбінований датасет: {len(df_combined)} записів")
    print(df_combined["label_id"].value_counts().sort_index().rename(LABEL_MAP))

    df_combined[["text", "label_id", "Label"]].to_csv(OUT_PATH, index=False)
    print(f"\nЗбережено: {OUT_PATH}")

if __name__ == "__main__":
    main()