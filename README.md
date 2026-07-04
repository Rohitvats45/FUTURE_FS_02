# 🎫 Support Ticket Classification System

An end-to-end NLP system that automatically classifies customer support tickets into
categories and assigns a priority level (**high / medium / low**), so support teams can
respond faster and smarter.

---

## 📁 Project Files

| File | Description |
|---|---|
| `Support_Ticket_Classification.ipynb` | Main Jupyter notebook — full pipeline from raw text to trained, evaluated models |
| `app.py` | Interactive Streamlit web app (classify single tickets, bulk classify a CSV, view model performance) |
| `sample_tickets.csv` | Synthetic labeled dataset (102 tickets, 6 categories, 3 priority levels) used to train/demo the models |
| `dummy_tickets.csv` | Unlabeled sample tickets for testing the app's bulk-classify / file-upload feature |

---

## 🔹 What It Does

1. **Text Cleaning & Tokenization** — lowercases text, strips URLs/punctuation, removes
   stopwords, and lemmatizes tokens (NLTK-based, with an automatic offline fallback if
   NLTK data can't be downloaded).
2. **Category Classification** — trains and compares three models (Logistic Regression,
   Linear SVM, Multinomial Naive Bayes) on TF-IDF features, and automatically selects
   the best performer.
3. **Priority Tagging** — a **hybrid** approach:
   - An ML classifier trained on TF-IDF text features + an engineered "urgency score"
   - A rule-based safety-net override that force-escalates any ticket containing 3+
     strong urgency keywords (e.g. *urgent, down, locked, deadline, escalate*) to **high**
4. **Model Evaluation** — accuracy, weighted F1, per-class precision/recall, and
   confusion matrices for both the category and priority models.
5. **End-to-End Pipeline** — a single `classify_ticket(text)` function that returns
   category, priority, and confidence scores for any new ticket.

---

## 🗂️ Categories & Priorities

**Categories:** Billing, Technical Issue, Account Access, Bug Report, Feature Request, General Inquiry

**Priority levels:** `high`, `medium`, `low`

---

## 🚀 Getting Started

### Option A — Jupyter Notebook

1. Make sure `Support_Ticket_Classification.ipynb` and `sample_tickets.csv` are in the
   same folder.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib nltk
   ```
3. Open the notebook in Jupyter or VS Code and **Run All** cells.
4. The notebook trains both models, shows evaluation charts, runs demo classifications,
   and saves the trained models as `.joblib` files for reuse.

### Option B — Streamlit App

1. Make sure `app.py` and `sample_tickets.csv` are in the same folder.
2. Install Streamlit (if not already installed):
   ```bash
   pip install streamlit
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Your browser will open the app automatically (`localhost:8501`). From there you can:
   - **Classify a Ticket** — type or paste a ticket, get instant category + priority
   - **Bulk Classify (CSV)** — upload `dummy_tickets.csv` (or your own) to classify many
     tickets at once and download the results
   - **Model Performance** — view accuracy/F1 scores and data distribution charts

---

## 🔄 Using Your Own Ticket Data

Replace `sample_tickets.csv` with your own export. Required columns:

| Column | Required for |
|---|---|
| `text` | Always required — the raw ticket content |
| `category` | Training the category model |
| `priority` | Training the priority model |

If you don't yet have historical `priority` labels, you can start by relying on the
rule-based `urgency_score` alone, then bootstrap the ML priority model once you've
accumulated agent-assigned labels.

---

## 🛠️ Customizing Urgency Detection

Priority tagging relies partly on a keyword-based `URGENCY_WORDS` set (in both
`app.py` and the notebook). You can tune this list to better match your domain —
just be careful not to add overly generic words (e.g. "help", "please"), as this can
cause **false positives** where normal tickets get marked high priority.

---

## 📈 Example Results (on the bundled sample data)

- **Category classification accuracy:** ~96%
- **Priority classification accuracy (hybrid):** ~88%

*(Results will vary once you train on your own real ticket data.)*

---

## 🔮 Possible Extensions

- SLA-based priority (e.g. auto-escalate for enterprise customers)
- Incorporate ticket metadata (customer tier, product area, past ticket count)
- Swap TF-IDF for embeddings (e.g. sentence-transformers) for better semantic matching
- Wrap `classify_ticket()` in a Flask/FastAPI service for real-time integration with a
  helpdesk tool

---

## 🧰 Tools Used

Python · NLTK · Scikit-learn · Pandas · Matplotlib/Seaborn · Streamlit
