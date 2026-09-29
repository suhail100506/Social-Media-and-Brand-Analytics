"""
Flask Application for Social Media Analytics Dashboard API.
Provides RESTful endpoints for metrics, topics, sentiment, engagement,
trends, spikes, post exploration, and real-time sentiment prediction.
"""
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from flask import Flask, request, jsonify
from flask_cors import CORS

from services.data_service import data_service
from services.prediction_service import prediction_service

app = Flask(__name__)

# Configure CORS
# Allow http://localhost:5173 (Vite default dev server) and configurable via environment
cors_origin = os.environ.get("CORS_ORIGIN", "http://localhost:5173")
CORS(app, resources={r"/api/*": {"origins": [cors_origin, "http://127.0.0.1:5173", "http://localhost:3000", "*"]}})


def get_query_filters():
    """Extract standard query filters from request arguments."""
    return {
        "topic": request.args.get("topic"),
        "platform": request.args.get("platform"),
        "sentiment": request.args.get("sentiment"),
        "start": request.args.get("start"),
        "end": request.args.get("end"),
    }


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------
@app.errorhandler(400)
def bad_request(error):
    msg = getattr(error, "description", "Bad request.")
    return jsonify({"error": msg, "status": 400}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found.", "status": 404}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "An internal server error occurred.", "status": 500}), 500


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint indicating whether data and models loaded successfully."""
    try:
        data = data_service.get_health(prediction_service.is_loaded())
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/summary", methods=["GET"])
def summary():
    """
    Get executive summary metrics.
    Supports optional query filters: ?topic=&platform=&sentiment=&start=&end=
    """
    try:
        filters = get_query_filters()
        data = data_service.get_summary(filters)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to compute summary: {str(e)}"}), 500


@app.route("/api/topics", methods=["GET"])
def topics():
    """
    Get per-topic aggregates: post_count, total_engagement, avg_engagement,
    positive_pct, negative_pct, neutral_pct.
    Supports optional filters: ?platform=&sentiment=&start=&end=
    """
    try:
        filters = get_query_filters()
        data = data_service.get_topics(filters)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve topic metrics: {str(e)}"}), 500


@app.route("/api/sentiment", methods=["GET"])
def sentiment():
    """
    Get overall sentiment breakdown + topic-vs-sentiment matrix + platform-vs-sentiment matrix.
    Supports optional filters: ?topic=&platform=&start=&end=
    """
    try:
        filters = get_query_filters()
        data = data_service.get_sentiment(filters)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve sentiment metrics: {str(e)}"}), 500


@app.route("/api/engagement", methods=["GET"])
def engagement():
    """
    Get engagement metrics per topic and per platform (total, average, post count).
    Supports optional filters: ?topic=&platform=&sentiment=&start=&end=
    """
    try:
        filters = get_query_filters()
        data = data_service.get_engagement(filters)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve engagement metrics: {str(e)}"}), 500


@app.route("/api/trends", methods=["GET"])
def trends():
    """
    Time series of post counts per topic over time.
    Query params:
        ?freq=D (daily, default) or W (weekly)
        ?topic=&platform=&sentiment=&start=&end=
    """
    try:
        filters = get_query_filters()
        freq = request.args.get("freq", "D").strip().upper()
        if freq not in ["D", "W"]:
            freq = "D"
        data = data_service.get_trends(filters, freq=freq)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to compute trends: {str(e)}"}), 500


@app.route("/api/spikes", methods=["GET"])
def spikes():
    """
    Get discussion spikes sorted by spike_ratio descending.
    Query params:
        ?min_ratio=2.0
        ?topic=&start=&end=
    """
    try:
        filters = get_query_filters()
        min_ratio_param = request.args.get("min_ratio")
        min_ratio = None
        if min_ratio_param:
            try:
                min_ratio = float(min_ratio_param)
            except ValueError:
                return jsonify({"error": "Invalid min_ratio parameter. Must be a numeric value."}), 400

        data = data_service.get_spikes(filters, min_ratio=min_ratio)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve spikes: {str(e)}"}), 500


@app.route("/api/posts", methods=["GET"])
def posts():
    """
    Get paginated and searchable posts.
    Query params:
        ?page=1
        ?limit=10
        ?q=keyword
        ?sort_by=date|engagement|likes
        ?sort_order=desc|asc
        ?topic=&platform=&sentiment=&start=&end=
    """
    try:
        filters = get_query_filters()
        q = request.args.get("q")
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 10))
        sort_by = request.args.get("sort_by", "date").lower()
        sort_order = request.args.get("sort_order", "desc").lower()

        data = data_service.get_posts(
            filters=filters,
            q=q,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return jsonify(data), 200
    except ValueError as ve:
        return jsonify({"error": f"Invalid pagination parameters: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve posts: {str(e)}"}), 500


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    """
    Get accuracy, precision, recall (reads metrics.json if provided, otherwise returns nulls gracefully).
    """
    try:
        metrics = prediction_service.get_model_metrics()
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve model metrics: {str(e)}"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict sentiment for a given social media post.
    Payload: {"text": "..."}
    Returns: {"sentiment": "positive", "confidence": 94.32, "probabilities": {...}}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON with a 'text' field."}), 400

        body = request.get_json(silent=True) or {}
        text = body.get("text")

        if text is None:
            return jsonify({"error": "Missing required field 'text' in JSON body."}), 400

        result = prediction_service.predict(text)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
