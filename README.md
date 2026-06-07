# ⚽ FIFA World Cup 2026 — AI Prediction & Analytics Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Predicted WC2026 Champion: Spain (16.77%)**

## Live Demo
[![Click Here](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fifa-wc2026-predictor.streamlit.app/)

---

## Overview
End-to-end Machine Learning project that predicts FIFA World Cup 2026 outcomes using:
- **49,378** historical international match results (1872–2024)
- **23 engineered features** including Elo ratings, recent form, H2H stats
- **4 ML models** trained and compared (Logistic Regression, Random Forest, XGBoost, Neural Network)
- **10,000 Monte Carlo simulations** of the full tournament bracket

## Architecture
```
Data Collection → EDA → Feature Engineering → Model Training → Simulation → Streamlit App
```

## Tech Stack
| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost, TensorFlow |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Data | Pandas, NumPy |

## Model Results
| Model | Accuracy | Log Loss | F1 Score |
|-------|----------|----------|----------|
| **Logistic Regression** | **59.64%** | **0.8692** | 0.4457 |
| Random Forest | 57.49% | 0.8879 | 0.5201 |
| XGBoost | 55.89% | 0.9016 | 0.5197 |
| Neural Network | 54.44% | 0.8983 | 0.5250 |

> 55-60% accuracy is considered very good for football prediction

## WC2026 Top Predictions
| Rank | Team | Win Probability |
|------|------|-----------------|
| 1st  | Spain | 16.77% |
| 2nd  | Argentina | 11.45% |
| 3rd  | France | 9.59% |

## Project Structure
```
fifa-wc2026-ai-prediction/
├── app.py                         # Streamlit web application
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── results.csv
│   └── processed/
│       ├── features.csv
│       ├── simulation_results.csv
│       └── wc2026_groups.csv
└── models/
    ├── logistic_regression.pkl
    ├── scaler.pkl
    ├── elo_ratings.pkl
    └── feature_cols.pkl
```

## Run Locally
```bash
git clone https://github.com/Abhirup2728/fifa-wc2026-ai-prediction
cd fifa-wc2026-ai-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Key Features of the App
- **Match Predictor** — select any 2 WC2026 teams, get win/draw/loss probabilities
- **Team Analytics** — individual team dashboard with ELO, form, group fixtures
- **Tournament Bracket** — predicted bracket from 10,000 simulations
- **EDA Insights** — key findings from 49,378 historical matches

## Author
**Abhirup Gumtya** | B.Tech CSE (AI & ML) 
- GitHub: https://github.com/Abhirup2728
- LinkedIn: https://linkedin.com/in/abhirupgumtya
- Portfolio: https://abhirup-gumtya-portfolio.netlify.app

## License
MIT License — free to use and modify.
