"""
Data Service for Social Media Analytics.
Loads CSV datasets once at startup and provides aggregation, filtering,
trend calculation, discussion spikes, and post browsing.
"""
import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from config import (
    ANALYSIS_READY_CSV,
    BRAND_TOPIC_SUMMARY_CSV,
    DISCUSSION_SPIKES_CSV,
    COL_POST_ID,
    COL_TIMESTAMP,
    COL_PLATFORM,
    COL_USER,
    COL_FOLLOWERS_COUNT,
    COL_VERIFIED,
    COL_TEXT,
    COL_HASHTAGS,
    COL_COUNTRY,
    COL_LIKES,
    COL_RETWEETS_SHARES,
    COL_COMMENTS,
    COL_SENTIMENT,
    COL_ENGAGEMENT,
    COL_DATE,
    COL_YEAR,
    COL_MONTH,
    COL_DAY,
    COL_HOUR,
    COL_TOPIC,
    COL_CLEAN_TEXT,
    COL_TOPIC_SUMMARY_TOPIC,
    COL_TOPIC_SUMMARY_POST_COUNT,
    COL_TOPIC_SUMMARY_TOTAL_ENGAGEMENT,
    COL_TOPIC_SUMMARY_AVERAGE_ENGAGEMENT,
    COL_TOPIC_SUMMARY_NEGATIVE_RATE,
    COL_TOPIC_SUMMARY_POSITIVE_RATE,
    COL_SPIKE_DATE,
    COL_SPIKE_TOPIC,
    COL_SPIKE_POST_COUNT,
    COL_SPIKE_ROLLING_AVERAGE,
    COL_SPIKE_RATIO,
    COL_SPIKE_FLAG,
    PLATFORM_MAPPING,
    SENTIMENT_POSITIVE,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    VALID_SENTIMENTS,
)


def sanitize_val(val: Any) -> Any:
    """Recursively replace float NaN and infinite values with None for JSON compliance."""
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, dict):
        return {k: sanitize_val(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_val(v) for v in val]
    if pd.isna(val):
        return None
    return val


class DataService:
    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.brand_summary_df: pd.DataFrame = pd.DataFrame()
        self.spikes_df: pd.DataFrame = pd.DataFrame()
        self.loaded: bool = False
        self._load_datasets()

    def _load_datasets(self):
        """Load and pre-process CSV datasets once at startup."""
        for path, name in [
            (ANALYSIS_READY_CSV, "social_media_analysis_ready.csv"),
            (BRAND_TOPIC_SUMMARY_CSV, "brand_topic_summary.csv"),
            (DISCUSSION_SPIKES_CSV, "discussion_spikes.csv"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required dataset {name} not found at {path}. "
                    "Please ensure it is located in backend/data/."
                )

        try:
            # 1. Main analysis ready dataframe
            df = pd.read_csv(ANALYSIS_READY_CSV)

            # Normalize platform names to canonical TitleCase (e.g. TWITTER -> Twitter)
            df[COL_PLATFORM] = df[COL_PLATFORM].astype(str).str.strip().str.lower().map(
                PLATFORM_MAPPING
            ).fillna(df[COL_PLATFORM].astype(str).str.strip().str.title())

            # Normalize sentiment to lowercase, preserving NaN
            df[COL_SENTIMENT] = df[COL_SENTIMENT].apply(
                lambda s: str(s).strip().lower() if pd.notna(s) and str(s).strip().lower() in VALID_SENTIMENTS else None
            )

            # Ensure numeric fields are well-formed
            df[COL_LIKES] = pd.to_numeric(df[COL_LIKES], errors="coerce").fillna(0).astype(int)
            df[COL_RETWEETS_SHARES] = pd.to_numeric(df[COL_RETWEETS_SHARES], errors="coerce").fillna(0).astype(int)
            df[COL_COMMENTS] = pd.to_numeric(df[COL_COMMENTS], errors="coerce").fillna(0).astype(float)
            df[COL_ENGAGEMENT] = pd.to_numeric(df[COL_ENGAGEMENT], errors="coerce").fillna(
                df[COL_LIKES] + df[COL_RETWEETS_SHARES] + df[COL_COMMENTS]
            ).astype(float)

            # Standardize Date string
            df[COL_DATE] = df[COL_DATE].astype(str).str.strip()
            df["_parsed_date"] = pd.to_datetime(df[COL_DATE], errors="coerce")

            # Clean followers count string (handle "1,125" etc.)
            def clean_followers(v):
                if pd.isna(v):
                    return None
                if isinstance(v, (int, float)):
                    return int(v) if v >= 0 else None
                s = str(v).replace(",", "").strip()
                try:
                    num = float(s)
                    return int(num) if num >= 0 else None
                except ValueError:
                    return None

            df[COL_FOLLOWERS_COUNT] = df[COL_FOLLOWERS_COUNT].apply(clean_followers)
            df[COL_TOPIC] = df[COL_TOPIC].astype(str).str.strip().fillna("Other")

            self.df = df

            # 2. Topic summary dataset
            summary_df = pd.read_csv(BRAND_TOPIC_SUMMARY_CSV)
            self.brand_summary_df = summary_df

            # 3. Spikes dataset
            spikes = pd.read_csv(DISCUSSION_SPIKES_CSV)
            spikes[COL_SPIKE_DATE] = spikes[COL_SPIKE_DATE].astype(str).str.strip()
            spikes["_parsed_date"] = pd.to_datetime(spikes[COL_SPIKE_DATE], errors="coerce")
            spikes[COL_SPIKE_TOPIC] = spikes[COL_SPIKE_TOPIC].astype(str).str.strip()
            spikes[COL_SPIKE_SPIKE_RATIO := COL_SPIKE_RATIO] = pd.to_numeric(
                spikes[COL_SPIKE_RATIO], errors="coerce"
            ).fillna(0.0)
            self.spikes_df = spikes

            self.loaded = True
            print(f"[DataService] Loaded {len(self.df)} posts, {len(self.brand_summary_df)} topic summaries, "
                  f"and {len(self.spikes_df)} spikes.")
        except Exception as e:
            self.loaded = False
            raise RuntimeError(f"Error loading datasets: {e}")

    def is_loaded(self) -> bool:
        return self.loaded and not self.df.empty

    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Filter posts dataframe based on optional query parameters."""
        filtered = df

        topic = filters.get("topic")
        if topic and topic.strip() and topic.lower() != "all":
            filtered = filtered[filtered[COL_TOPIC].str.lower() == topic.strip().lower()]

        platform = filters.get("platform")
        if platform and platform.strip() and platform.lower() != "all":
            filtered = filtered[filtered[COL_PLATFORM].str.lower() == platform.strip().lower()]

        sentiment = filters.get("sentiment")
        if sentiment and sentiment.strip() and sentiment.lower() != "all":
            filtered = filtered[filtered[COL_SENTIMENT].str.lower() == sentiment.strip().lower()]

        start_date = filters.get("start")
        if start_date and start_date.strip():
            filtered = filtered[filtered[COL_DATE] >= start_date.strip()]

        end_date = filters.get("end")
        if end_date and end_date.strip():
            filtered = filtered[filtered[COL_DATE] <= end_date.strip()]

        return filtered

    def get_health(self, prediction_loaded: bool) -> Dict[str, Any]:
        """System health and asset loading status."""
        return sanitize_val({
            "status": "healthy" if (self.loaded and prediction_loaded) else "degraded",
            "models_loaded": prediction_loaded,
            "data_loaded": self.loaded,
            "total_posts": len(self.df),
            "total_spikes": len(self.spikes_df),
            "total_topics": len(self.brand_summary_df),
            "date_range": {
                "start": self.df[COL_DATE].min() if not self.df.empty else None,
                "end": self.df[COL_DATE].max() if not self.df.empty else None,
            }
        })

    def get_summary(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute top-level summary metrics.
        Returns:
            total_posts, total_engagement, avg_engagement, positive/negative/neutral percentages,
            most_discussed_topic, highest_engagement_topic, most_negative_topic, most_positive_topic,
            most_active_platform, spike_count.
        """
        filtered = self.apply_filters(self.df, filters)
        total_posts = len(filtered)

        if total_posts == 0:
            return sanitize_val({
                "total_posts": 0,
                "total_engagement": 0,
                "avg_engagement": 0.0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "sentiment_counts": {"positive": 0, "negative": 0, "neutral": 0},
                "most_discussed_topic": None,
                "highest_engagement_topic": None,
                "most_negative_topic": None,
                "most_positive_topic": None,
                "most_active_platform": None,
                "spike_count": 0,
            })

        total_engagement = float(filtered[COL_ENGAGEMENT].sum())
        avg_engagement = round(float(filtered[COL_ENGAGEMENT].mean()), 2)

        # Sentiment counts
        sent_counts = filtered[COL_SENTIMENT].value_counts()
        pos_cnt = int(sent_counts.get(SENTIMENT_POSITIVE, 0))
        neg_cnt = int(sent_counts.get(SENTIMENT_NEGATIVE, 0))
        neu_cnt = int(sent_counts.get(SENTIMENT_NEUTRAL, 0))

        # Percentages relative to total posts in selection
        pos_pct = round((pos_cnt / total_posts) * 100.0, 2)
        neg_pct = round((neg_cnt / total_posts) * 100.0, 2)
        neu_pct = round((neu_cnt / total_posts) * 100.0, 2)

        # Topics analytics
        topic_counts = filtered[COL_TOPIC].value_counts()
        most_discussed_topic = topic_counts.index[0] if not topic_counts.empty else None

        topic_eng_sum = filtered.groupby(COL_TOPIC)[COL_ENGAGEMENT].sum()
        highest_engagement_topic = topic_eng_sum.idxmax() if not topic_eng_sum.empty else None

        # Per topic sentiment rates for most_negative / most_positive
        topic_grouped = filtered.groupby(COL_TOPIC)
        topic_stats = []
        for name, group in topic_grouped:
            cnt = len(group)
            t_neg = (group[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum()
            t_pos = (group[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum()
            topic_stats.append({
                "topic": name,
                "neg_rate": (t_neg / cnt) * 100.0 if cnt > 0 else 0,
                "pos_rate": (t_pos / cnt) * 100.0 if cnt > 0 else 0,
            })

        if topic_stats:
            most_negative_topic = max(topic_stats, key=lambda x: x["neg_rate"])["topic"]
            most_positive_topic = max(topic_stats, key=lambda x: x["pos_rate"])["topic"]
        else:
            most_negative_topic = None
            most_positive_topic = None

        # Most active platform
        plat_counts = filtered[COL_PLATFORM].value_counts()
        most_active_platform = plat_counts.index[0] if not plat_counts.empty else None

        # Spikes count matching date/topic filters
        spikes_f = self.spikes_df
        t_filter = filters.get("topic")
        if t_filter and t_filter.lower() != "all":
            spikes_f = spikes_f[spikes_f[COL_SPIKE_TOPIC].str.lower() == t_filter.strip().lower()]
        start_d = filters.get("start")
        if start_d:
            spikes_f = spikes_f[spikes_f[COL_SPIKE_DATE] >= start_d.strip()]
        end_d = filters.get("end")
        if end_d:
            spikes_f = spikes_f[spikes_f[COL_SPIKE_DATE] <= end_d.strip()]
        spike_count = len(spikes_f)

        return sanitize_val({
            "total_posts": total_posts,
            "total_engagement": total_engagement,
            "avg_engagement": avg_engagement,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "sentiment_counts": {
                "positive": pos_cnt,
                "negative": neg_cnt,
                "neutral": neu_cnt,
            },
            "most_discussed_topic": most_discussed_topic,
            "highest_engagement_topic": highest_engagement_topic,
            "most_negative_topic": most_negative_topic,
            "most_positive_topic": most_positive_topic,
            "most_active_platform": most_active_platform,
            "spike_count": spike_count,
        })

    def get_topics(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Per-topic breakdown of post count, engagement, and sentiment percentages.
        Dynamically adapts to platform/date/sentiment filters.
        """
        filtered = self.apply_filters(self.df, filters)
        if filtered.empty:
            return []

        results = []
        for topic, group in filtered.groupby(COL_TOPIC):
            cnt = len(group)
            tot_eng = float(group[COL_ENGAGEMENT].sum())
            avg_eng = round(tot_eng / cnt, 2) if cnt > 0 else 0.0

            pos = (group[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum()
            neg = (group[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum()
            neu = (group[COL_SENTIMENT] == SENTIMENT_NEUTRAL).sum()

            results.append({
                "topic": topic,
                "post_count": cnt,
                "total_engagement": tot_eng,
                "avg_engagement": avg_eng,
                "positive_pct": round((pos / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "negative_pct": round((neg / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "neutral_pct": round((neu / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "positive_count": int(pos),
                "negative_count": int(neg),
                "neutral_count": int(neu),
            })

        # Sort by post count descending
        results.sort(key=lambda x: x["post_count"], reverse=True)
        return sanitize_val(results)

    def get_sentiment(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overall sentiment breakdown, Topic vs Sentiment matrix, and Platform vs Sentiment matrix.
        """
        filtered = self.apply_filters(self.df, filters)
        total_posts = len(filtered)

        overall = {
            "positive": int((filtered[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum()),
            "negative": int((filtered[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum()),
            "neutral": int((filtered[COL_SENTIMENT] == SENTIMENT_NEUTRAL).sum()),
            "unlabeled": int(filtered[COL_SENTIMENT].isna().sum()),
            "total": total_posts,
            "positive_pct": round(((filtered[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum() / total_posts * 100.0), 2) if total_posts > 0 else 0.0,
            "negative_pct": round(((filtered[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum() / total_posts * 100.0), 2) if total_posts > 0 else 0.0,
            "neutral_pct": round(((filtered[COL_SENTIMENT] == SENTIMENT_NEUTRAL).sum() / total_posts * 100.0), 2) if total_posts > 0 else 0.0,
        }

        # Topic vs Sentiment
        topic_vs_sentiment = []
        for topic, group in filtered.groupby(COL_TOPIC):
            cnt = len(group)
            pos = int((group[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum())
            neg = int((group[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum())
            neu = int((group[COL_SENTIMENT] == SENTIMENT_NEUTRAL).sum())
            topic_vs_sentiment.append({
                "topic": topic,
                "total": cnt,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "positive_pct": round((pos / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "negative_pct": round((neg / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "neutral_pct": round((neu / cnt) * 100.0, 2) if cnt > 0 else 0.0,
            })
        topic_vs_sentiment.sort(key=lambda x: x["total"], reverse=True)

        # Platform vs Sentiment
        platform_vs_sentiment = []
        for platform, group in filtered.groupby(COL_PLATFORM):
            cnt = len(group)
            pos = int((group[COL_SENTIMENT] == SENTIMENT_POSITIVE).sum())
            neg = int((group[COL_SENTIMENT] == SENTIMENT_NEGATIVE).sum())
            neu = int((group[COL_SENTIMENT] == SENTIMENT_NEUTRAL).sum())
            platform_vs_sentiment.append({
                "platform": platform,
                "total": cnt,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "positive_pct": round((pos / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "negative_pct": round((neg / cnt) * 100.0, 2) if cnt > 0 else 0.0,
                "neutral_pct": round((neu / cnt) * 100.0, 2) if cnt > 0 else 0.0,
            })
        platform_vs_sentiment.sort(key=lambda x: x["total"], reverse=True)

        return sanitize_val({
            "overall": overall,
            "topic_vs_sentiment": topic_vs_sentiment,
            "platform_vs_sentiment": platform_vs_sentiment
        })

    def get_engagement(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Engagement statistics per topic and per platform (total, average, post count).
        """
        filtered = self.apply_filters(self.df, filters)

        by_topic = []
        for topic, group in filtered.groupby(COL_TOPIC):
            cnt = len(group)
            tot_eng = float(group[COL_ENGAGEMENT].sum())
            avg_eng = round(tot_eng / cnt, 2) if cnt > 0 else 0.0
            tot_likes = int(group[COL_LIKES].sum())
            tot_shares = int(group[COL_RETWEETS_SHARES].sum())
            tot_comments = int(group[COL_COMMENTS].sum())

            by_topic.append({
                "topic": topic,
                "post_count": cnt,
                "total_engagement": tot_eng,
                "avg_engagement": avg_eng,
                "total_likes": tot_likes,
                "total_shares": tot_shares,
                "total_comments": tot_comments,
            })
        by_topic.sort(key=lambda x: x["total_engagement"], reverse=True)

        by_platform = []
        for platform, group in filtered.groupby(COL_PLATFORM):
            cnt = len(group)
            tot_eng = float(group[COL_ENGAGEMENT].sum())
            avg_eng = round(tot_eng / cnt, 2) if cnt > 0 else 0.0
            tot_likes = int(group[COL_LIKES].sum())
            tot_shares = int(group[COL_RETWEETS_SHARES].sum())
            tot_comments = int(group[COL_COMMENTS].sum())

            by_platform.append({
                "platform": platform,
                "post_count": cnt,
                "total_engagement": tot_eng,
                "avg_engagement": avg_eng,
                "total_likes": tot_likes,
                "total_shares": tot_shares,
                "total_comments": tot_comments,
            })
        by_platform.sort(key=lambda x: x["total_engagement"], reverse=True)

        return sanitize_val({
            "by_topic": by_topic,
            "by_platform": by_platform
        })

    def get_trends(self, filters: Dict[str, Any], freq: str = "D") -> List[Dict[str, Any]]:
        """
        Time series aggregation of post count per topic.
        Supports daily (D) or weekly (W) grouping.
        """
        filtered = self.apply_filters(self.df, filters).copy()
        if filtered.empty:
            return []

        all_topics = sorted(self.df[COL_TOPIC].unique().tolist())

        if freq.upper() == "W":
            # Group by weekly period starting Monday
            filtered["period"] = filtered["_parsed_date"].dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
        else:
            # Default daily
            filtered["period"] = filtered[COL_DATE]

        # Pivot: period x Topic -> count
        pivot = pd.crosstab(filtered["period"], filtered[COL_TOPIC])

        # Ensure all known topics exist as columns
        for t in all_topics:
            if t not in pivot.columns:
                pivot[t] = 0

        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.reset_index().rename(columns={"period": "date"}).sort_values("date")

        records = pivot.to_dict(orient="records")
        return sanitize_val(records)

    def get_spikes(self, filters: Dict[str, Any], min_ratio: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Discussion spikes sorted by spike ratio descending with optional filtering.
        """
        spikes = self.spikes_df.copy()

        topic = filters.get("topic")
        if topic and topic.strip() and topic.lower() != "all":
            spikes = spikes[spikes[COL_SPIKE_TOPIC].str.lower() == topic.strip().lower()]

        start_date = filters.get("start")
        if start_date and start_date.strip():
            spikes = spikes[spikes[COL_SPIKE_DATE] >= start_date.strip()]

        end_date = filters.get("end")
        if end_date and end_date.strip():
            spikes = spikes[spikes[COL_SPIKE_DATE] <= end_date.strip()]

        if min_ratio is not None and min_ratio > 0:
            spikes = spikes[spikes[COL_SPIKE_RATIO] >= float(min_ratio)]

        spikes = spikes.sort_values(COL_SPIKE_RATIO, ascending=False)

        clean_records = []
        for _, row in spikes.iterrows():
            clean_records.append({
                "date": str(row[COL_SPIKE_DATE]),
                "topic": str(row[COL_SPIKE_TOPIC]),
                "post_count": int(row[COL_SPIKE_POST_COUNT]),
                "rolling_average": round(float(row[COL_SPIKE_ROLLING_AVERAGE]), 2),
                "spike_ratio": round(float(row[COL_SPIKE_RATIO]), 2),
                "spike": bool(row[COL_SPIKE_FLAG]),
            })

        return sanitize_val(clean_records)

    def get_posts(
        self,
        filters: Dict[str, Any],
        q: Optional[str] = None,
        page: int = 1,
        limit: int = 10,
        sort_by: str = "date",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        Search and paginate posts.
        """
        filtered = self.apply_filters(self.df, filters)

        # Keyword search
        if q and q.strip():
            query = q.strip().lower()
            text_match = filtered[COL_TEXT].str.lower().str.contains(query, na=False)
            user_match = filtered[COL_USER].str.lower().str.contains(query, na=False)
            tag_match = filtered[COL_HASHTAGS].astype(str).str.lower().str.contains(query, na=False)
            country_match = filtered[COL_COUNTRY].astype(str).str.lower().str.contains(query, na=False)
            filtered = filtered[text_match | user_match | tag_match | country_match]

        # Sorting
        if sort_by == "engagement":
            filtered = filtered.sort_values(COL_ENGAGEMENT, ascending=(sort_order == "asc"))
        elif sort_by == "likes":
            filtered = filtered.sort_values(COL_LIKES, ascending=(sort_order == "asc"))
        else:
            # Default timestamp / date sorting
            filtered = filtered.sort_values(COL_TIMESTAMP, ascending=(sort_order == "asc"))

        total = len(filtered)
        page = max(1, page)
        limit = max(1, min(100, limit))
        total_pages = math.ceil(total / limit) if limit > 0 else 1

        offset = (page - 1) * limit
        page_df = filtered.iloc[offset : offset + limit].copy()

        # Format records for JSON response
        posts_list = []
        for _, row in page_df.iterrows():
            posts_list.append({
                "post_id": str(row[COL_POST_ID]),
                "timestamp": str(row[COL_TIMESTAMP]),
                "date": str(row[COL_DATE]),
                "platform": str(row[COL_PLATFORM]),
                "user": str(row[COL_USER]),
                "followers_count": sanitize_val(row[COL_FOLLOWERS_COUNT]),
                "verified": str(row[COL_VERIFIED]) if pd.notna(row[COL_VERIFIED]) else "No",
                "text": str(row[COL_TEXT]),
                "hashtags": str(row[COL_HASHTAGS]) if pd.notna(row[COL_HASHTAGS]) else "",
                "country": str(row[COL_COUNTRY]) if pd.notna(row[COL_COUNTRY]) else "",
                "likes": int(row[COL_LIKES]),
                "shares": int(row[COL_RETWEETS_SHARES]),  # Normalized key for frontend
                "retweets_shares": int(row[COL_RETWEETS_SHARES]),
                "comments": int(row[COL_COMMENTS]),
                "sentiment": row[COL_SENTIMENT] if pd.notna(row[COL_SENTIMENT]) else None,
                "engagement": float(row[COL_ENGAGEMENT]),
                "topic": str(row[COL_TOPIC]),
            })

        return sanitize_val({
            "posts": posts_list,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        })


# Singleton instance
data_service = DataService()
