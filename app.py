import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import finnhub
import streamlit as st

# Import functions from your core file
from newsPull import analyze, get_relevant_articles, summarize_results

st.title("Financial News Sentiment Analyzer")

ticker = st.text_input("Enter company ticker (e.g. AAPL, TSLA):", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", datetime.today() - timedelta(days=7))
with col2:
    end_date = st.date_input("End date", datetime.today())

if st.button("Analyze"):
    api = st.secrets["FINNHUB"]["API_KEY"]
    finnhub_client = finnhub.Client(api_key=api)

    news = finnhub_client.company_news(
        ticker,
        _from=start_date.strftime('%Y-%m-%d'),
        to=end_date.strftime('%Y-%m-%d')
    )

    relevant_news = get_relevant_articles(news, ticker)
    if not relevant_news:
        st.warning("No relevant news articles found.")
    else:
        results = analyze(relevant_news)
        summary, percentages, score = summarize_results(results)

        # Summary
        st.subheader("Summary Statistics")
        st.write(summary)
        st.write({k: f"{v:.1f}%" for k, v in percentages.items()})
        st.write(f"Overall Sentiment Score: {score:.2f}")

        # Chart
        fig, ax = plt.subplots()
        ax.bar(summary.keys(), summary.values(), color=['green', 'red', 'blue', 'gray'])
        ax.set_ylabel("Number of Articles")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        # Article Results
        st.subheader("Article Sentiments")
        for r in results[:10]:
            st.markdown(f"**{r['headline']}**")
            st.write(f"Sentiment: {r['sentiment']}")
            st.write(
                f"Scores → Positive: {r['scores']['positive']}, "
                f"Negative: {r['scores']['negative']}, "
                f"Neutral: {r['scores']['neutral']}"
            )
            st.write(f"[Read more]({r['url']})")
            st.write("---")
