# Predicting AI Model Elo Ratings

A data science project that predicts a language model's **Chatbot Arena Elo rating**
from its metadata (vote count, rating variance, developer organization, and license),
using the full historical LMSYS/Chatbot Arena leaderboard (May 2023 – July 2026).

## Project Overview

Each row in the dataset is a model's ranking on a published leaderboard snapshot.
The goal is a **regression** model that predicts the `rating` (Elo score) — a measure
of human-preference quality — from features that don't leak the answer.

| Step | What happens |
|------|--------------|
| 1. Load & explore | Inspect 97k rows, data types, and missing values |
| 2. Clean | Parse dates, fill missing labels, handle early-snapshot gaps |
| 3. Visualize (EDA) | Correlation heatmap and vote-count vs. rating scatter |
| 4. Model | Linear Regression baseline vs. Random Forest |
| 5. Evaluate | R², RMSE, feature importance, predicted-vs-actual plot |

## Key Results

- **Random Forest:** R² = **0.71**, RMSE ≈ **81 Elo points**
- Big improvement over the Linear Regression baseline
- **Most predictive features:** rating variance, developer organization, and vote count
- **Takeaway:** rating stability and popularity meaningfully predict model quality,
  but ~29% of the variation remains unexplained — quality isn't fully reducible to metadata.

>  **Avoiding leakage:** `rank`, `rating_lower`, and `rating_upper` were deliberately
> excluded from the model — they are derived directly from the rating and would give an
> artificially perfect score.

## Repository Contents

| File | Description |
|------|-------------|
| `AI Ranking DS project.ipynb` | The full analysis notebook |
| `ai_model_arena_rankings.csv.csv` | Raw dataset (all snapshots, all arenas) |
| `latest_text_leaderboard.csv` | Cleaned modeling dataset (latest text leaderboard) |
| `arena_rating_model.pk1` | Trained Random Forest model (load with `joblib.load`) |
| `requirements.txt` | Python dependencies |

## Images
<img width="1480" height="805" alt="frontier" src="https://github.com/user-attachments/assets/b0968fd0-15ed-4ba8-bddd-33d8f931f333" />
<img width="1479" height="881" alt="org_dominance" src="https://github.com/user-attachments/assets/6f252f9e-50a4-49ce-b49f-80841a8042c9" />
<img width="1481" height="805" alt="top_models_race" src="https://github.com/user-attachments/assets/d701dd46-1614-46dc-a21b-4707f18e006c" />

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/EmmanuelV26/ai-model-rating-prediction.git
cd ai-model-rating-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook "AI Ranking DS project.ipynb"
```

## Dataset

AI model leaderboard rankings covering 5 arenas (text, text-style-control, vision,
webdev, search), with Elo ratings, confidence intervals, vote counts, and ranks for
every model at every published snapshot since May 2023.
Can be Found on Kaggle

## Built With

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn
