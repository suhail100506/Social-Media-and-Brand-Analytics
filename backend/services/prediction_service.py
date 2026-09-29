"""
Prediction Service for Social Media Analytics.
Loads the trained Multinomial Naive Bayes model and TF-IDF vectorizer once at startup.
Handles text preprocessing, sentiment inference, probability calculation, and metrics loading.
"""
import json
import warnings
from typing import Dict, Any, Optional

import joblib
import numpy as np

from config import (
    SENTIMENT_MODEL_PKL,
    TFIDF_VECTORIZER_PKL,
    METRICS_JSON,
    SENTIMENT_POSITIVE,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    VALID_SENTIMENTS
)
from utils.text_cleaner import clean_text


class PredictionService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.classes_ = []
        self.loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and vectorizer once at startup with clear error reporting."""
        if not SENTIMENT_MODEL_PKL.exists():
            raise FileNotFoundError(
                f"Sentiment model pickle not found at {SENTIMENT_MODEL_PKL}. "
                "Ensure naive_bayes_sentiment_model.pkl is placed in backend/models/."
            )
        if not TFIDF_VECTORIZER_PKL.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer pickle not found at {TFIDF_VECTORIZER_PKL}. "
                "Ensure tfidf_vectorizer.pkl is placed in backend/models/."
            )

        try:
            with warnings.catch_warnings(record=True) as recorded_warnings:
                warnings.simplefilter("always")
                self.model = joblib.load(SENTIMENT_MODEL_PKL)
                self.vectorizer = joblib.load(TFIDF_VECTORIZER_PKL)
                for w in recorded_warnings:
                    print(f"[Model Loader Warning] {w.category.__name__}: {w.message}")

            self.classes_ = [str(c).lower() for c in getattr(self.model, "classes_", [])]
            self.loaded = True
            print(f"[PredictionService] Successfully loaded model ({type(self.model).__name__}) "
                  f"and vectorizer ({len(self.vectorizer.vocabulary_)} terms). Classes: {self.classes_}")
        except Exception as e:
            self.loaded = False
            raise RuntimeError(f"Failed to load ML artifacts: {str(e)}")

    def is_loaded(self) -> bool:
        return self.loaded and self.model is not None and self.vectorizer is not None

    def predict(self, text: Any) -> Dict[str, Any]:
        """
        Predict sentiment for given text input.
        
        Args:
            text: Raw post text string.
            
        Returns:
            dict containing sentiment, confidence (0-100), probabilities, and cleaned text.
            
        Raises:
            ValueError: If text is empty, not a string, or exceeds 1000 characters.
        """
        if text is None or not isinstance(text, str):
            raise ValueError("Input 'text' must be a non-empty string.")
        
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input 'text' cannot be empty or whitespace only.")
        
        if len(text) > 1000:
            raise ValueError(f"Input text exceeds maximum allowed length of 1000 characters (length: {len(text)}).")

        cleaned = clean_text(stripped)
        
        # Transform using fitted TF-IDF vectorizer
        # If all words were stopwords, TF-IDF vector will be zero vector, which MultinomialNB handles via priors.
        X = self.vectorizer.transform([cleaned if cleaned else stripped.lower()])
        raw_probs = self.model.predict_proba(X)[0]

        # Map raw probabilities to valid sentiment classes
        # Model classes include ['nan', 'negative', 'neutral', 'positive']
        # We extract only positive, negative, neutral and normalize them over their sum
        class_prob_map = dict(zip(self.classes_, raw_probs))
        
        p_pos = float(class_prob_map.get(SENTIMENT_POSITIVE, 0.0))
        p_neg = float(class_prob_map.get(SENTIMENT_NEGATIVE, 0.0))
        p_neu = float(class_prob_map.get(SENTIMENT_NEUTRAL, 0.0))

        valid_sum = p_pos + p_neg + p_neu
        if valid_sum > 0:
            norm_pos = (p_pos / valid_sum) * 100.0
            norm_neg = (p_neg / valid_sum) * 100.0
            norm_neu = (p_neu / valid_sum) * 100.0
        else:
            norm_pos = 33.33
            norm_neg = 33.33
            norm_neu = 33.34

        prob_dict = {
            SENTIMENT_POSITIVE: round(norm_pos, 2),
            SENTIMENT_NEGATIVE: round(norm_neg, 2),
            SENTIMENT_NEUTRAL: round(norm_neu, 2)
        }

        # Select highest probability among valid sentiments
        best_sentiment = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_sentiment]

        return {
            "sentiment": best_sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
            "cleaned_text": cleaned
        }

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Load model metrics from metrics.json if provided, otherwise return nulls gracefully.
        """
        if METRICS_JSON.exists():
            try:
                with open(METRICS_JSON, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                    return {
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1_score": metrics.get("f1_score", metrics.get("f1")),
                        "available": True,
                        "source": "metrics.json"
                    }
            except Exception as e:
                print(f"[PredictionService] Warning: Failed to parse {METRICS_JSON}: {e}")

        # Return null values gracefully as required by spec
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "available": False,
            "message": "metrics.json not provided yet. Metrics return null gracefully."
        }


# Singleton instance
prediction_service = PredictionService()
