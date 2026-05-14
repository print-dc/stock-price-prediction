"""
app/app.py
Flask dashboard for real-time stock prediction visualisation.

Endpoints
---------
GET  /                        → Main dashboard HTML
GET  /api/predict/<ticker>    → Latest prediction JSON
GET  /api/history/<ticker>    → 90-day close history
GET  /api/batch/<ticker>      → Actual vs predicted (test set)
GET  /api/metrics/<ticker>    → Saved training metrics
GET  /api/tickers             → List of trained tickers
"""

import os
import sys
import json
import glob

from flask import Flask, jsonify, render_template, abort
from flask_cors import CORS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _trained_tickers():
    """Return list of tickers that have a saved model."""
    pattern = os.path.join(ARTIFACTS_DIR, "*_best.keras")
    paths   = glob.glob(pattern)
    return [os.path.basename(p).replace("_best.keras", "") for p in paths]


def _load_metrics(ticker: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, f"{ticker.upper()}_metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_history(ticker: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, f"{ticker.upper()}_history.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tickers")
def api_tickers():
    return jsonify({"tickers": _trained_tickers()})


@app.route("/api/predict/<ticker>")
def api_predict(ticker: str):
    try:
        from model.predict import predict_latest
        data = predict_latest(ticker.upper())
        return jsonify(data)
    except FileNotFoundError as e:
        abort(404, description=str(e))
    except Exception as e:
        abort(500, description=str(e))


@app.route("/api/batch/<ticker>")
def api_batch(ticker: str):
    try:
        from model.predict import predict_batch
        data = predict_batch(ticker.upper())
        return jsonify(data)
    except FileNotFoundError as e:
        abort(404, description=str(e))
    except Exception as e:
        abort(500, description=str(e))


@app.route("/api/metrics/<ticker>")
def api_metrics(ticker: str):
    metrics = _load_metrics(ticker)
    history = _load_history(ticker)
    if not metrics:
        abort(404, description=f"No metrics found for {ticker}. Train the model first.")
    return jsonify({"metrics": metrics, "history": history})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e.description)}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": str(e.description)}), 500


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
