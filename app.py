"""
Support Ticket Classification - Streamlit App
-----------------------------------------------
Run with:  streamlit run app.py

Trains (and caches) a category + priority classifier on sample_tickets.csv,
then lets you classify new tickets interactively, or upload a CSV of
tickets to classify in bulk.
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------
# NLTK with graceful offline fallback (same logic as the notebook)
# ---------------------------------------------------------------
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    STOPWORDS = set(nltk_stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()

    def lemmatize(token):
        return _lemmatizer.lemmatize(token)

except Exception:
    STOPWORDS = set("""a an the is are was were be been being to of in on for and or but if
    then so than that this these those i you he she it we they me him her us them my your
    his its our their as at by with from up down out about into over under again further
    do does did doing have has had having not no nor can will just should now""".split())

    def lemmatize(token):
        for suf in ("ing", "ies", "ed", "es", "s"):
            if token.endswith(suf) and len(token) > len(suf) + 2:
                return token[: -len(suf)]
        return token


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatize(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


URGENCY_WORDS = {
    "urgent", "immediately", "asap", "now", "critical", "right", "cannot", "cant",
    "broken", "down", "locked", "lost", "deadline", "escalate", "emergency",
}


def urgency_score(raw_text: str) -> int:
    t = str(raw_text).lower()
    return sum(1 for w in URGENCY_WORDS if w in t)


# ---------------------------------------------------------------
# Train (cached so it only runs once per session / data change)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train_pipeline(df: pd.DataFrame):
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df["urgency_score"] = df["text"].apply(urgency_score)

    # ---- category model ----
    train_df, test_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["category"]
    )
    cat_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_cat = cat_vectorizer.fit_transform(train_df["clean_text"])
    X_test_cat = cat_vectorizer.transform(test_df["clean_text"])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    cat_results = {}
    for name, model in candidates.items():
        model.fit(X_train_cat, train_df["category"])
        preds = model.predict(X_test_cat)
        cat_results[name] = {
            "model": model,
            "accuracy": accuracy_score(test_df["category"], preds),
            "f1": f1_score(test_df["category"], preds, average="weighted"),
            "report": classification_report(test_df["category"], preds, output_dict=True),
        }
    best_cat_name = max(cat_results, key=lambda k: cat_results[k]["f1"])
    best_cat_model = cat_results[best_cat_name]["model"]

    # ---- priority model (hybrid) ----
    train_df2, test_df2 = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["priority"]
    )
    pri_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_text = pri_vectorizer.fit_transform(train_df2["clean_text"])
    X_test_text = pri_vectorizer.transform(test_df2["clean_text"])
    X_train_pri = hstack([X_train_text, train_df2[["urgency_score"]].values])
    X_test_pri = hstack([X_test_text, test_df2[["urgency_score"]].values])

    priority_model = LogisticRegression(max_iter=1000)
    priority_model.fit(X_train_pri, train_df2["priority"])
    raw_preds = priority_model.predict(X_test_pri)
    final_preds = [
        "high" if urgency_score(t) >= 3 else p
        for t, p in zip(test_df2["text"], raw_preds)
    ]
    pri_accuracy = accuracy_score(test_df2["priority"], final_preds)
    pri_report = classification_report(test_df2["priority"], final_preds, output_dict=True)

    return {
        "cat_vectorizer": cat_vectorizer,
        "cat_model": best_cat_model,
        "best_cat_name": best_cat_name,
        "cat_results": cat_results,
        "pri_vectorizer": pri_vectorizer,
        "pri_model": priority_model,
        "pri_accuracy": pri_accuracy,
        "pri_report": pri_report,
        "df": df,
    }


def classify_ticket(text: str, pipeline: dict) -> dict:
    cleaned = clean_text(text)
    u_score = urgency_score(text)

    cat_vec = pipeline["cat_vectorizer"].transform([cleaned])
    category = pipeline["cat_model"].predict(cat_vec)[0]
    if hasattr(pipeline["cat_model"], "predict_proba"):
        cat_conf = float(pipeline["cat_model"].predict_proba(cat_vec).max())
    else:
        cat_conf = None

    pri_vec = pipeline["pri_vectorizer"].transform([cleaned])
    pri_full = hstack([pri_vec, np.array([[u_score]])])
    priority = pipeline["pri_model"].predict(pri_full)[0]
    pri_conf = float(pipeline["pri_model"].predict_proba(pri_full).max())

    if u_score >= 3:
        priority = "high"

    return {
        "category": category,
        "category_confidence": cat_conf,
        "priority": priority,
        "priority_confidence": pri_conf,
        "urgency_score": u_score,
    }


# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.set_page_config(page_title="Ticket Classifier", page_icon="🎫", layout="wide")
st.title("🎫 Support Ticket Classification")
st.caption("Automatic category + priority tagging for support tickets, powered by TF-IDF + scikit-learn.")

with st.sidebar:
    st.header("Training Data")
    uploaded = st.file_uploader("Upload your own labeled CSV (needs: text, category, priority)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} tickets from upload.")
    else:
        df = pd.read_csv("sample_tickets.csv")
        st.info(f"Using bundled sample_tickets.csv ({len(df)} tickets).")

pipeline = train_pipeline(df)

tab1, tab2, tab3 = st.tabs(["🔍 Classify a Ticket", "📊 Bulk Classify (CSV)", "📈 Model Performance"])

with tab1:
    st.subheader("Classify a single ticket")
    example = st.selectbox(
        "Try an example, or write your own below:",
        [
            "",
            "Our production server is down and no customers can check out, please help immediately!",
            "Just wondering how to change my display name.",
            "I was charged twice for my subscription this month, can someone check the invoice?",
        ],
    )
    text_input = st.text_area("Ticket text", value=example, height=120,
                               placeholder="Paste or type a support ticket here...")

    if st.button("Classify", type="primary"):
        if text_input.strip():
            result = classify_ticket(text_input, pipeline)
            col1, col2, col3 = st.columns(3)
            col1.metric("Category", result["category"],
                        f"{result['category_confidence']:.0%} conf." if result["category_confidence"] else None)
            priority_color = {"high": "🔴", "medium": "🟠", "low": "🟢"}
            col2.metric("Priority", f"{priority_color.get(result['priority'],'')} {result['priority']}",
                        f"{result['priority_confidence']:.0%} conf.")
            col3.metric("Urgency keywords found", result["urgency_score"])
        else:
            st.warning("Please enter some ticket text first.")

with tab2:
    st.subheader("Classify many tickets at once")
    bulk_file = st.file_uploader("Upload a CSV with a `text` column", type=["csv"], key="bulk")
    if bulk_file is not None:
        bulk_df = pd.read_csv(bulk_file)
        if "text" not in bulk_df.columns:
            st.error("CSV must have a `text` column.")
        else:
            results = bulk_df["text"].apply(lambda t: pd.Series(classify_ticket(t, pipeline)))
            out_df = pd.concat([bulk_df, results], axis=1)
            st.dataframe(out_df, use_container_width=True)
            st.download_button("Download results as CSV", out_df.to_csv(index=False),
                                file_name="classified_tickets.csv", mime="text/csv")

with tab3:
    st.subheader("Category model comparison")
    cat_summary = pd.DataFrame({
        name: {"Accuracy": r["accuracy"], "Weighted F1": r["f1"]}
        for name, r in pipeline["cat_results"].items()
    }).T
    st.dataframe(cat_summary.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Selected model: **{pipeline['best_cat_name']}**")

    st.subheader("Priority model (hybrid ML + rules)")
    st.metric("Accuracy", f"{pipeline['pri_accuracy']:.1%}")
    pri_report_df = pd.DataFrame(pipeline["pri_report"]).T
    st.dataframe(pri_report_df.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Training data distribution")
    col1, col2 = st.columns(2)
    col1.bar_chart(pipeline["df"]["category"].value_counts())
    col2.bar_chart(pipeline["df"]["priority"].value_counts())
