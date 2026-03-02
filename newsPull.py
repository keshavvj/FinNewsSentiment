import os
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from newspaper import Article
from collections import Counter
import streamlit as st


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, model = load_model()


def get_api_key():
    """Fetch Finnhub API key from Streamlit secrets or environment variable."""
    try:
        return st.secrets["FINNHUB"]["API_KEY"]
    except Exception:
        key = os.environ.get("FINNHUB_API_KEY")
        if not key:
            raise ValueError(
                "Finnhub API key not found. Set it via .streamlit/secrets.toml "
                "or the FINNHUB_API_KEY environment variable."
            )
        return key





def analyze(n):
    sentiment_results = []
    labels = ["Positive", "Negative", "Neutral"]

    for article in n:
        full_text = get_full_text(article['url'])

        if not full_text or len(full_text.strip()) < 100:
            full_text = article['headline'] + " " + article['summary']

        # Split into ≤512-token chunks
        tokens = tokenizer(full_text, return_tensors="pt", truncation=False, padding=False)
        input_ids = tokens["input_ids"][0]
        
        # Break into chunks of max 512 tokens
        chunk_size = 512
        chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
        
        # Run each chunk separately
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

        # Average probabilities across chunks
        avg_probs = np.mean(chunk_probs, axis=0)
        prediction = int(np.argmax(avg_probs))
        confidence = float(max(avg_probs))

        if confidence < 0.6:
            sentiment = "Uncertain"
        else:
            sentiment = labels[prediction]

        sentiment_results.append({
            'headline': article['headline'],
            'sentiment': sentiment,
            'url': article['url'],
            'scores': {
                'positive': round(avg_probs[0], 3),
                'negative': round(avg_probs[1], 3),
                'neutral': round(avg_probs[2], 3)
            }
        })

    return sentiment_results

def summarize_results(results):
    counts = Counter([r['sentiment'] for r in results])
    total = sum(counts.values())
    summary = {label: counts.get(label, 0) for label in ["Positive", "Negative", "Neutral", "Uncertain"]}
    percentages = {k: (v / total * 100 if total > 0 else 0) for k, v in summary.items()}
    score = (summary.get("Positive", 0) - summary.get("Negative", 0)) / total if total > 0 else 0
    return summary, percentages, score

def get_relevant_articles(news_list, company):
    company_upper = company.upper()
    filtered = []
    for article in news_list:
        combined_text = (article.get('headline', '') + ' ' + article.get('summary', '')).upper()
        if company_upper in combined_text:
            filtered.append(article)
    return filtered

def get_full_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        
        # Filter out paragraphs shorter than 50 characters
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
        cleaned_text = ' '.join(paragraphs)
        
        return cleaned_text if cleaned_text else None
    except Exception as e:
        print(f"Error extracting article: {e}")
        return None
