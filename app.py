from datetime import datetime, timedelta

import finnhub
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from newsPull import (
    analyze,
    daily_sentiment,
    get_api_key,
    get_relevant_articles,
    lexicon_baseline,
    results_to_frame,
    summarize_results,
)


SENTIMENT_COLORS = {
    "Positive": "#2e7d32",
    "Negative": "#c62828",
    "Neutral": "#1565c0",
    "Uncertain": "#6d6d6d",
}
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]


st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    layout="wide",
)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_company_news(api_key, ticker, start_date, end_date):
    client = finnhub.Client(api_key=api_key)
    return client.company_news(
        ticker,
        _from=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_snapshot(api_key, ticker):
    client = finnhub.Client(api_key=api_key)
    profile = client.company_profile2(symbol=ticker)
    quote = client.quote(ticker)
    return profile or {}, quote or {}


def plot_distribution(summary):
    labels = list(summary.keys())
    counts = list(summary.values())
    colors = [SENTIMENT_COLORS.get(label, "#6d6d6d") for label in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel("Articles")
    ax.set_title("Sentiment Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_daily_trend(daily_frame):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not daily_frame.empty:
        for ticker, group in daily_frame.groupby("ticker"):
            group = group.sort_values("date")
            ax.plot(
                pd.to_datetime(group["date"]),
                group["sentiment_score"],
                marker="o",
                linewidth=2,
                label=ticker,
            )
    ax.axhline(0, color="#777777", linewidth=1, alpha=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Average sentiment score")
    ax.set_title("Daily Sentiment Dynamics")
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def render_market_cards(api_key, tickers):
    cols = st.columns(min(len(tickers), 4))
    for index, ticker in enumerate(tickers):
        with cols[index % len(cols)]:
            try:
                profile, quote = fetch_market_snapshot(api_key, ticker)
                current = quote.get("c")
                change = quote.get("d")
                percent = quote.get("dp")
                name = profile.get("name") or ticker
                metric_label = f"{ticker} - {name}"
                metric_value = f"${current:,.2f}" if isinstance(current, (int, float)) else "N/A"
                metric_delta = (
                    f"{change:+.2f} ({percent:+.2f}%)"
                    if isinstance(change, (int, float))
                    and isinstance(percent, (int, float))
                    else None
                )
                st.metric(metric_label, metric_value, metric_delta)
            except Exception as exc:
                st.warning(f"{ticker}: quote unavailable ({exc})")


def render_article_table(results_frame):
    display = results_frame.copy()
    display["published_at"] = pd.to_datetime(display["published_at"], errors="coerce")
    display["published_at"] = display["published_at"].dt.strftime("%Y-%m-%d %H:%M")
    display = display[
        [
            "ticker",
            "published_at",
            "source",
            "sentiment",
            "confidence",
            "headline",
            "url",
        ]
    ]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url", display_text="Open"),
            "confidence": st.column_config.NumberColumn("confidence", format="%.3f"),
        },
    )


st.title("Financial News Sentiment Analyzer")

with st.sidebar:
    st.header("Analysis Setup")
    selected_tickers = st.multiselect(
        "Major stock tickers",
        DEFAULT_TICKERS,
        default=["AAPL", "MSFT", "TSLA"],
    )
    custom_tickers = st.text_input("Additional tickers", placeholder="AMD, NFLX, JPM")
    start_date = st.date_input("Start date", datetime.today() - timedelta(days=7))
    end_date = st.date_input("End date", datetime.today())
    max_articles = st.slider("Max articles per ticker", 5, 100, 25, step=5)
    include_full_text = st.checkbox(
        "Download full article text",
        value=False,
        help="Slower, but can improve context when publishers allow article parsing.",
    )
    analyze_clicked = st.button("Run sentiment analysis", type="primary")

custom_list = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
tickers = sorted({ticker.upper() for ticker in [*selected_tickers, *custom_list]})

if not tickers:
    st.info("Choose at least one ticker in the sidebar.")
    st.stop()

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()

try:
    api_key = get_api_key()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    "FinBERT sentiment classification over real-time Finnhub company news and market quotes."
)
render_market_cards(api_key, tickers)

if not analyze_clicked:
    st.info("Configure the sidebar and run sentiment analysis.")
    st.stop()

all_results = []
progress = st.progress(0, text="Fetching news and running FinBERT...")

for index, ticker in enumerate(tickers, start=1):
    progress.progress((index - 1) / len(tickers), text=f"Analyzing {ticker}...")
    news = fetch_company_news(api_key, ticker, start_date, end_date)
    relevant_news = get_relevant_articles(news, ticker)
    ticker_results = analyze(
        relevant_news,
        ticker=ticker,
        include_full_text=include_full_text,
        max_articles=max_articles,
    )
    all_results.extend(ticker_results)

progress.progress(1.0, text="Analysis complete.")

if not all_results:
    st.warning("No analyzable news articles were found for the selected tickers/date range.")
    st.stop()

summary, percentages, score = summarize_results(all_results)
results_frame = results_to_frame(all_results)
daily_frame = daily_sentiment(all_results)
baseline_frame = lexicon_baseline(all_results)

overview_tab, trend_tab, articles_tab, comparison_tab = st.tabs(
    ["Overview", "Sentiment Dynamics", "Articles", "Model Comparison"]
)

with overview_tab:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Articles analyzed", len(all_results))
    metric_cols[1].metric("Overall sentiment score", f"{score:+.2f}")
    metric_cols[2].metric("Positive share", f"{percentages['Positive']:.1f}%")
    metric_cols[3].metric("Negative share", f"{percentages['Negative']:.1f}%")

    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        st.pyplot(plot_distribution(summary), clear_figure=True)
    with table_col:
        st.dataframe(
            pd.DataFrame(
                {
                    "sentiment": summary.keys(),
                    "articles": summary.values(),
                    "share": [f"{percentages[label]:.1f}%" for label in summary.keys()],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with trend_tab:
    st.pyplot(plot_daily_trend(daily_frame), clear_figure=True)
    if not daily_frame.empty:
        st.dataframe(daily_frame, use_container_width=True, hide_index=True)

with articles_tab:
    render_article_table(results_frame)

with comparison_tab:
    matches = baseline_frame["match"].mean() if not baseline_frame.empty else 0
    st.metric("Rule baseline agreement with FinBERT", f"{matches * 100:.1f}%")
    st.dataframe(
        baseline_frame[["ticker", "finbert", "baseline", "match", "headline"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "The baseline is intentionally simple and is included only as a transparent "
        "reference point. A fine-tuned model can be added by replacing MODEL_NAME in "
        "newsPull.py with a local or hosted Hugging Face checkpoint."
    )
