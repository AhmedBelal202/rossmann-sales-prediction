# Rossmann Store Sales Prediction

Predicting daily sales of Rossmann drugstores with machine learning (XGBoost), based on the
[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) Kaggle competition data.

The project covers the full workflow: data loading, EDA, preprocessing, model training,
evaluation (hold-out + cross-validation), hyperparameter tuning, and generating predictions.

## Project Structure

```
rossmann-sales-prediction/
├── data/                  # train.csv, store.csv, test.csv
├── models/                # trained model (model.pkl) is saved here
├── outputs/
│   ├── figures/           # EDA plots
│   ├── metrics.json       # evaluation results
│   ├── best_params.json   # tuning results
│   └── submission.csv     # predictions for test.csv
├── rossmann.py            # all the code: EDA, training, tuning, prediction
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd rossmann-sales-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure `train.csv`, `store.csv` and `test.csv` are inside the `data/` folder.

## How to Run

Run all commands from the project root. All the code lives in a single file, `rossmann.py`, with four commands: `eda`, `train`, `tune`, `predict`.

| Step | Command | Output |
|------|---------|--------|
| 1. EDA | `python rossmann.py eda` | plots in `outputs/figures/` |
| 2. Train + evaluate | `python rossmann.py train` | `models/model.pkl`, `outputs/metrics.json` |
| 2b. Train with 5-fold CV | `python rossmann.py train --cv` | adds CV RMSE to the metrics |
| 2c. Train with extra features | `python rossmann.py train --extended` | see below |
| 3. Tune hyperparameters | `python rossmann.py tune --sample 100000` | `outputs/best_params.json` |
| 4. Predict on test set | `python rossmann.py predict` | `outputs/submission.csv` |

### Feature sets

- **Baseline (default):** store-level features only - `StoreType`, `Assortment`, `CompetitionDistance`,
  `CompetitionOpenSinceMonth/Year`, `Promo2`, `Promo2SinceWeek/Year`, `PromoInterval`.
- **Extended (`--extended`):** baseline + daily features - `DayOfWeek`, `Promo`, `SchoolHoliday`,
  `StateHoliday`, `Month`, `Year`, `WeekOfYear`. These usually improve accuracy a lot, since sales
  depend heavily on the day and on whether a promotion is running.

## Method

1. **Data:** `train.csv` is merged with `store.csv` on `Store`; rows with zero sales are removed.
2. **Preprocessing:** numeric features -> mean imputation; categorical features -> constant imputation
   (`Unknown`) + one-hot encoding. Everything lives in one scikit-learn `Pipeline`, so the exact same
   steps are applied at training and prediction time.
3. **Model:** `XGBRegressor` (100 trees, learning rate 0.1).
4. **Evaluation:** 80/20 hold-out split (R², MAE, RMSE) and optional 5-fold cross-validation.
5. **Tuning:** `RandomizedSearchCV` over a `RandomForestRegressor` pipeline (3-fold CV, 20 iterations).

## Results

Fill these in after running the scripts (values are saved in `outputs/metrics.json`).

| Feature set | R² | MAE | RMSE | CV RMSE (5-fold) |
|-------------|----|-----|------|------------------|
| Baseline    |    |     |      |                  |
| Extended    |    |     |      |                  |

EDA plots are available in `outputs/figures/`.

## Notes and Limitations

- The hold-out split and cross-validation are **random**, but the data is a time series. A time-based
  split (train on earlier dates, validate on later ones) would give a more realistic estimate.
- The baseline model only knows store attributes, so it predicts roughly the same value for every day of
  a given store. Use `--extended` for day-level predictions.
- Stores that are closed (`Open == 0`) in `test.csv` are predicted as 0 sales.
- The Kaggle competition metric is RMSPE, not RMSE; it is not computed here.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly.
