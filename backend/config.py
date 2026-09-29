"""
Configuration module for Social Media Analytics backend.
All column names and dataset/model path configurations are centralized here.
"""
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"

# Dataset Files
ANALYSIS_READY_CSV = DATA_DIR / "social_media_analysis_ready.csv"
BRAND_TOPIC_SUMMARY_CSV = DATA_DIR / "brand_topic_summary.csv"
DISCUSSION_SPIKES_CSV = DATA_DIR / "discussion_spikes.csv"
CLEANED_CSV = DATA_DIR / "social_media_sentiment_cleaned.csv"

# Model Files
SENTIMENT_MODEL_PKL = MODELS_DIR / "naive_bayes_sentiment_model.pkl"
TFIDF_VECTORIZER_PKL = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_JSON = MODELS_DIR / "metrics.json"

# Column names in social_media_analysis_ready.csv
COL_POST_ID = "post_id"
COL_TIMESTAMP = "timestamp"
COL_PLATFORM = "platform"
COL_USER = "user"
COL_FOLLOWERS_COUNT = "followers_count"
COL_VERIFIED = "verified"
COL_TEXT = "text"
COL_HASHTAGS = "hashtags"
COL_COUNTRY = "country"
COL_LIKES = "likes"
COL_RETWEETS_SHARES = "retweets_shares"  # Note: Actual column name in CSV is retweets_shares
COL_COMMENTS = "comments"
COL_SENTIMENT = "sentiment"
COL_ENGAGEMENT = "Engagement"
COL_DATE = "Date"
COL_YEAR = "Year"
COL_MONTH = "Month"
COL_DAY = "Day"
COL_HOUR = "Hour"
COL_CLEAN_TEXT = "Clean_Text"
COL_TOPIC = "Topic"

# Column names in brand_topic_summary.csv
COL_TOPIC_SUMMARY_TOPIC = "Topic"
COL_TOPIC_SUMMARY_POST_COUNT = "Post_Count"
COL_TOPIC_SUMMARY_TOTAL_ENGAGEMENT = "Total_Engagement"
COL_TOPIC_SUMMARY_AVERAGE_ENGAGEMENT = "Average_Engagement"
COL_TOPIC_SUMMARY_NEGATIVE_RATE = "Negative_Rate"
COL_TOPIC_SUMMARY_POSITIVE_RATE = "Positive_Rate"

# Column names in discussion_spikes.csv
COL_SPIKE_DATE = "Date"
COL_SPIKE_TOPIC = "Topic"
COL_SPIKE_POST_COUNT = "Post_Count"
COL_SPIKE_ROLLING_AVERAGE = "Rolling_Average"
COL_SPIKE_RATIO = "Spike_Ratio"
COL_SPIKE_FLAG = "Spike"

# Sentiment values
SENTIMENT_POSITIVE = "positive"
SENTIMENT_NEGATIVE = "negative"
SENTIMENT_NEUTRAL = "neutral"
VALID_SENTIMENTS = [SENTIMENT_POSITIVE, SENTIMENT_NEGATIVE, SENTIMENT_NEUTRAL]

# Platform canonical mapping (to handle uppercase/mixed case in source CSV)
PLATFORM_MAPPING = {
    "twitter": "Twitter",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "linkedin": "LinkedIn"
}
