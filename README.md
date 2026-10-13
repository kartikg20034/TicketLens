# TicketLens 🔎
### Semantic Support Ticket Retrieval & Classification

**Find similar support tickets. Understand issues faster.**

TicketLens is a Python CLI tool that uses semantic search to retrieve similar customer-support tickets and recommend an issue category. It combines pretrained BGE embeddings, similarity-based retrieval, and a TF-IDF + Logistic Regression baseline.

## ✨ Features

- **Semantic search** — Find tickets with similar meaning using BGE embeddings.
- **Category recommendation** — Generate top-1 and top-5 majority-category recommendations.
- **Explainable results** — View similar tickets, similarity scores, and the rationale behind recommendations.
- **ML baseline** — Compare embedding-based retrieval against TF-IDF + Logistic Regression.
- **Data integrity checks** — Validate dataset balance, split distribution, and scenario-group isolation.
- **CLI-first design** — No frontend required.

## 🛠️ Tech Stack

`Python 3.12` · `Sentence Transformers` · `BAAI/bge-base-en-v1.5` · `scikit-learn` · `pandas` · `pytest` · `Supabase / pgvector` (supporting experiment infrastructure)

## ⚙️ How It Works

1. Enter a ticket description in the terminal.
2. Generate an embedding using the pretrained BGE model.
3. Retrieve semantically similar tickets from the training set.
4. Recommend a category based on retrieved examples.
5. Compare results with a TF-IDF + Logistic Regression baseline.

## 📊 Dataset & Results

The experiment uses **1,000 synthetic support tickets** across five categories.

| Split | Tickets |
|---|---:|
| Training | 700 |
| Validation | 150 |
| Test | 150 |
| **Total** | **1,000** |

Categories: Billing inquiry, Cancellation request, Product inquiry, Refund request, and Technical issue.

### Held-out test results

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| BGE top-1 retrieval | 82.67% | 0.8264 |
| BGE top-5 majority | 80.00% | 0.7903 |
| TF-IDF + Logistic Regression | 80.00% | 0.7913 |

**Key findings**
- BGE top-1 retrieval outperformed the baseline by 2.67 percentage points in test accuracy.
- Test performance was lower than validation performance, indicating a generalization gap.
- Technical issues were frequently confused with product inquiries.
- Top-5 majority voting did not improve test performance over top-1 retrieval.

These results apply to the synthetic dataset and are not evidence of production performance.

## 📁 Project Structure

```text
ticketlens/
├── backend/             # Supporting and legacy demo modules
├── data/
│   └── processed/       # Synthetic ticket dataset
├── scripts/
│   ├── route_synthetic_ticket.py
│   ├── evaluate_synthetic_routing.py
│   └── inspect_synthetic_integrity.py
├── supabase/
│   └── migrations/      # Supporting database experiment
├── tests/               # Automated tests
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

Requires Python 3.12.

```bash
git clone <your-repository-url>
cd ticketlens

python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Check the dataset:

```bash
python scripts/inspect_synthetic_integrity.py
```

Try a ticket:

```bash
printf '%s\n' 'The app crashes when I open the receipt screen.' \
  | python scripts/route_synthetic_ticket.py
```

Evaluate on the validation split:

```bash
python scripts/evaluate_synthetic_routing.py --split validation
```

Run the automated tests:

```bash
python -m pytest
```

**Last recorded test result:** 67 tests passed.

## ⚠️ Current Scope & Limitations

- The dataset is synthetic and may not represent real-world support-ticket distributions.
- The system recommends one of five issue categories; it does **not** perform verified engineering-team assignment.
- Similarity scores measure embedding similarity, not calibrated prediction confidence.
- Billing, refunds, cancellations, and technical issues can overlap.
- Output redaction covers common sensitive-data patterns but cannot guarantee complete removal of personal information.

## 🔮 Future Enhancements

- **Engineering-team routing:** Add validated team labels and explicit ownership rules.
- **Human-in-the-loop review:** Capture corrections and flag ambiguous tickets.
- **Confidence-aware recommendations:** Identify uncertain predictions and defer to human review.
- **Improved generalization:** Expand evaluation data and investigate recurring classification errors.
- **Ticketing integrations:** Connect to support platforms after validating routing quality.

## 🔒 Data & License

The experiment uses synthetic data inspired by public support workflows; it is not represented as genuine customer data or as affiliated with the referenced companies. Keep credentials and sensitive datasets out of version control.

No open-source license is provided for this project. Third-party data and dependencies remain subject to their applicable terms.

---

**TicketLens** explores how semantic retrieval can make support-ticket triage faster by surfacing relevant historical examples and providing explainable category recommendations.
