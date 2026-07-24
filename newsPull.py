import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
from newspaper import Article
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "yiyanghkust/finbert-tone"
SENTIMENT_LABELS = ("Positive", "Negative", "Neutral")
UNCERTAIN_LABEL = "Uncertain"
CONFIDENCE_THRESHOLD = 0.6


@st.cache_resource(show_spinner=False)
def load_model(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_api_key():
    """Fetch Finnhub API key from Streamlit secrets or an environment variable."""
    try:
        return st.secrets["FINNHUB"]["API_KEY"]
    except Exception:
        key = os.environ.get("FINNHUB_API_KEY")
        if not key:
            raise ValueError(
                "Finnhub API key not found. Set .streamlit/secrets.toml or "
                "the FINNHUB_API_KEY environment variable."
            )
        return key


def get_relevant_articles(news_list, ticker):
    """Keep articles that mention the ticker in the headline or summary."""
    ticker_upper = ticker.upper()
    filtered = []

    for article in news_list:
        combined_text = (
            f"{article.get('headline', '')} {article.get('summary', '')}"
        ).upper()
        if ticker_upper in combined_text:
            filtered.append(article)

    return filtered or news_list


def get_full_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        paragraphs = [p.strip() for p in article.text.split("\n") if len(p.strip()) > 50]
        return " ".join(paragraphs) or None
    except Exception:
        return None


def _normalize_label(raw_label):
    label = str(raw_label).strip().lower()
    if "positive" in label:
        return "Positive"
    if "negative" in label:
        return "Negative"
    if "neutral" in label:
        return "Neutral"
    return str(raw_label).strip().title()


def _label_order(model):
    labels = []
    for index in range(model.config.num_labels):
        labels.append(_normalize_label(model.config.id2label.get(index, index)))
    return labels


def _article_text(article, include_full_text):
    headline = article.get("headline", "")
    summary = article.get("summary", "")
    fallback = f"{headline}. {summary}".strip()

    if include_full_text and article.get("url"):
        full_text = get_full_text(article["url"])
        if full_text and len(full_text.strip()) >= 100:
            return full_text

    return fallback


def _predict_text(text, tokenizer, model):
    labels = _label_order(model)
    tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens["input_ids"][0]
    chunk_size = min(tokenizer.model_max_length, 512)
    chunks = [input_ids[i : i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

    chunk_probs = []
    for chunk in chunks:
        chunk_2d = chunk.unsqueeze(0)
        inputs = {
            "input_ids": chunk_2d,
            "attention_mask": torch.ones_like(chunk_2d),
        }
        with torch.no_grad():
            outputs = model(**inputs)
            probs_tensor = torch.nn.functional.softmax(outputs.logits, dim=1)
            chunk_probs.append(probs_tensor.squeeze().tolist())

    avg_probs = np.mean(chunk_probs, axis=0)
    prediction = int(np.argmax(avg_probs))
    confidence = float(max(avg_probs))
    sentiment = labels[prediction] if confidence >= CONFIDENCE_THRESHOLD else UNCERTAIN_LABEL

    scores = {label.lower(): 0.0 for label in SENTIMENT_LABELS}
    for label, probability in zip(labels, avg_probs):
        if label in SENTIMENT_LABELS:
            scores[label.lower()] = round(float(probability), 3)

    return sentiment, scores, confidence


def analyze(articles, ticker=None, include_full_text=False, max_articles=50):
    tokenizer, model = load_model()
    sentiment_results = []

    for article in articles[:max_articles]:
        text = _article_text(article, include_full_text)
        if not text:
            continue

        sentiment, scores, confidence = _predict_text(text, tokenizer, model)
        timestamp = article.get("datetime")
        published_at = (
            datetime.fromtimestamp(timestamp)
            if isinstance(timestamp, (int, float)) and timestamp > 0
            else None
        )

        sentiment_results.append(
            {
                "ticker": ticker,
                "headline": article.get("headline", "Untitled article"),
                "summary": article.get("summary", ""),
                "source": article.get("source", "Unknown"),
                "published_at": published_at,
                "sentiment": sentiment,
                "confidence": round(confidence, 3),
                "url": article.get("url", ""),
                "scores": scores,
            }
        )

    return sentiment_results


def summarize_results(results):
    counts = Counter(r["sentiment"] for r in results)
    total = sum(counts.values())
    summary = {
        label: counts.get(label, 0)
        for label in (*SENTIMENT_LABELS, UNCERTAIN_LABEL)
    }
    percentages = {k: (v / total * 100 if total else 0) for k, v in summary.items()}
    score = (summary["Positive"] - summary["Negative"]) / total if total else 0
    return summary, percentages, score


def results_to_frame(results):
    rows = []
    sentiment_values = {
        "Positive": 1.0,
        "Neutral": 0.0,
        "Negative": -1.0,
        "Uncertain": 0.0,
    }

    for result in results:
        rows.append(
            {
                "ticker": result.get("ticker"),
                "published_at": result.get("published_at"),
                "date": result.get("published_at").date()
                if result.get("published_at")
                else None,
                "headline": result.get("headline"),
                "source": result.get("source"),
                "sentiment": result.get("sentiment"),
                "confidence": result.get("confidence"),
                "sentiment_value": sentiment_values.get(result.get("sentiment"), 0.0),
                "positive": result.get("scores", {}).get("positive", 0.0),
                "negative": result.get("scores", {}).get("negative", 0.0),
                "neutral": result.get("scores", {}).get("neutral", 0.0),
                "url": result.get("url"),
            }
        )

    return pd.DataFrame(rows)


def daily_sentiment(results):
    frame = results_to_frame(results)
    if frame.empty or "date" not in frame:
        return frame

    return (
        frame.dropna(subset=["date"])
        .groupby(["ticker", "date"], as_index=False)
        .agg(
            article_count=("headline", "count"),
            sentiment_score=("sentiment_value", "mean"),
            positive=("positive", "mean"),
            negative=("negative", "mean"),
            neutral=("neutral", "mean"),
        )
    )


def lexicon_baseline(results):
    """A small transparent baseline for comparison with FinBERT."""
    positive_words = {
        "beat",
        "beats",
        "bullish",
        "gain",
        "gains",
        "growth",
        "outperform",
        "profit",
        "profits",
        "record",
        "surge",
        "upgraded",
    }
    negative_words = {
        "bearish",
        "cut",
        "decline",
        "downgraded",
        "drop",
        "falls",
        "lawsuit",
        "loss",
        "miss",
        "risk",
        "slump",
        "weak",
    }

    rows = []
    for result in results:
        words = {
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in result.get("headline", "").split()
        }
        positive_hits = len(words & positive_words)
        negative_hits = len(words & negative_words)
        if positive_hits > negative_hits:
            baseline = "Positive"
        elif negative_hits > positive_hits:
            baseline = "Negative"
        else:
            baseline = "Neutral"

        rows.append(
            {
                "ticker": result.get("ticker"),
                "headline": result.get("headline"),
                "finbert": result.get("sentiment"),
                "baseline": baseline,
                "match": baseline == result.get("sentiment"),
            }
        )

    return pd.DataFrame(rows)
