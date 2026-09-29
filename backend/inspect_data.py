import os
import joblib
import pandas as pd
import numpy as np
import warnings

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

print("=" * 80)
print("INSPECTING DATA FILES IN:", DATA_DIR)
print("=" * 80)

csv_files = [
    'brand_topic_summary.csv',
    'discussion_spikes.csv',
    'social_media_analysis_ready.csv',
    'social_media_sentiment_cleaned.csv',
    'social_media_sentiment_raw.csv'
]

for filename in csv_files:
    filepath = os.path.join(DATA_DIR, filename)
    print("\n" + "-" * 50)
    print(f"FILE: {filename}")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        continue
    df = pd.read_csv(filepath)
    print(f"Row count: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("Data types:")
    print(df.dtypes)
    print("\nHead(5):")
    print(df.head(5))

print("\n" + "=" * 80)
print("INSPECTING MODEL FILES IN:", MODELS_DIR)
print("=" * 80)

model_path = os.path.join(MODELS_DIR, 'naive_bayes_sentiment_model.pkl')
vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = joblib.load(model_path)
    if w:
        print(f"WARNING loading model: {w[-1].message}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    vectorizer = joblib.load(vec_path)
    if w:
        print(f"WARNING loading vectorizer: {w[-1].message}")

print(f"Model type: {type(model)}")
print(f"Model classes: {getattr(model, 'classes_', None)}")
print(f"Vectorizer type: {type(vectorizer)}")
vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 'N/A'
print(f"Vectorizer vocabulary size: {vocab_size}")

# Test sample prediction
test_text = ["This product launch exceeded all my expectations, fantastic work.", "Worst brand experience I've had in a long time, frustrating."]
X_test = vectorizer.transform(test_text)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
print(f"Sample test predictions: {preds}")
print(f"Sample test probabilities: {probs}")

# Inspect raw vs cleaned text
raw_df = pd.read_csv(os.path.join(DATA_DIR, 'social_media_sentiment_raw.csv'))
cleaned_df = pd.read_csv(os.path.join(DATA_DIR, 'social_media_sentiment_cleaned.csv'))
analysis_df = pd.read_csv(os.path.join(DATA_DIR, 'social_media_analysis_ready.csv'))

print("\n" + "=" * 80)
print("COMPARING RAW VS CLEANED TEXT (First 10 non-null):")
print("=" * 80)
for i in range(min(10, len(raw_df))):
    raw_t = raw_df.iloc[i].get('text', '')
    clean_t = cleaned_df.iloc[i].get('text', '') if 'text' in cleaned_df.columns else cleaned_df.iloc[i].get('Clean_Text', '')
    print(f"Raw:     {raw_t}")
    print(f"Cleaned: {clean_t}")
    print()
