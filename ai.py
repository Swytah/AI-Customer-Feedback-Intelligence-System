import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import pipeline

# Load dataset
df = pd.read_csv("sentiment_analysis.csv")

st.title(" AI Customer Feedback Analyzer")

st.write("Sample Data:")
st.write(df.head())

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["clean_tweet"] = df["tweet"].apply(clean_text)

# -----------------------------
# A/B TESTING MODELS
# -----------------------------

# Model A: TextBlob
def sentiment_textblob(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "POSITIVE"
    elif analysis.sentiment.polarity < 0:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# Model B: Transformer
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

transformer_model = load_model()

def sentiment_transformer(text):
    result = transformer_model(str(text)[:512])[0]
    return result["label"]

# Apply both models
df["Sentiment_A_TextBlob"] = df["clean_tweet"].apply(sentiment_textblob)
df["Sentiment_B_Transformer"] = df["clean_tweet"].apply(sentiment_transformer)

# Use Transformer as final output
df["Sentiment"] = df["Sentiment_B_Transformer"]

st.subheader("Sentiment Results")
st.write(df.head())

# -----------------------------
# A/B TESTING COMPARISON
# -----------------------------
st.subheader("🧪 A/B Testing Comparison")

agreement = (df["Sentiment_A_TextBlob"] == df["Sentiment_B_Transformer"])
agreement_rate = agreement.mean()

st.write(f"Model Agreement Rate: {agreement_rate:.2f}")

# -----------------------------
# ISSUE DETECTION
# -----------------------------
custom_stopwords = {
    "iphone", "apple", "samsung", "sony", "ipad",
    "phone", "just", "like", "love", "follow"
}

issue_keywords = [
    "bad", "worst", "slow", "problem", "issue",
    "broken", "poor", "error", "fail", "crash",
    "lag", "delay", "bug", "not working"
]

negative_reviews = df[df["Sentiment"] == "NEGATIVE"]

filtered_reviews = negative_reviews[
    negative_reviews["clean_tweet"].str.contains("|".join(issue_keywords), na=False)
]
all_words = " ".join(filtered_reviews["clean_tweet"]).split()

filtered_words = [
    word for word in all_words
    if word not in ENGLISH_STOP_WORDS
    and word not in custom_stopwords
    and len(word) > 3
]


meaningful_words = [
    word for word in filtered_words
    if word in issue_keywords
]


word_counts = Counter(meaningful_words)
common_words = word_counts.most_common(10)

st.subheader("Top Issues")
st.write(common_words)

# -----------------------------
# DASHBOARD
# -----------------------------
sentiment_counts = df["Sentiment"].value_counts()

st.subheader(" Sentiment Distribution")
st.bar_chart(sentiment_counts)

issues_df = pd.DataFrame(common_words, columns=["Issue", "Count"])

st.subheader(" Top Issues Chart")
st.bar_chart(issues_df.set_index("Issue"))

# -----------------------------
# AI GENERATED INSIGHTS
# -----------------------------
st.subheader(" AI-Generated Insights")

if common_words:
    top_issue = common_words[0][0]
    top_count = common_words[0][1]

    st.write(f" Most common issue: {top_issue} ({top_count} mentions)")

    st.write(" Key Insights:")
    for word, count in common_words[:5]:
        st.write(f"- {word} appears {count} times in negative feedback")

    negative_percentage = (df["Sentiment"] == "NEGATIVE").mean() * 100
    st.write(f"️ {negative_percentage:.1f}% of feedback is negative")

else:
    st.write("No major issues detected")
