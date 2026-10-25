"""
Rossmann Store Sales Prediction - single-file project.

Usage (run from the project root):
    python rossmann.py eda                      # exploratory data analysis -> outputs/figures/
    python rossmann.py train                    # train + evaluate -> models/model.pkl, outputs/metrics.json
    python rossmann.py train --cv               # + 5-fold cross-validation
    python rossmann.py train --extended         # + Promo, DayOfWeek, holidays, date features
    python rossmann.py tune --sample 100000     # hyperparameter tuning -> outputs/best_params.json
    python rossmann.py predict                  # predictions for data/test.csv -> outputs/submission.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "model.pkl"

RANDOM_STATE = 42
TARGET = "Sales"
REFERENCE_YEAR = 2015  # last year in the dataset (fixed so EDA results are reproducible)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Baseline features (store-level only)
BASE_NUMERIC = [
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
]
BASE_CATEGORICAL = ["StoreType", "Assortment", "PromoInterval"]
# Extra daily features, enabled with --extended
EXTRA_NUMERIC = ["DayOfWeek", "Promo", "SchoolHoliday", "Month", "Year", "WeekOfYear"]
EXTRA_CATEGORICAL = ["StateHoliday"]


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #
def read_csv(name, **kwargs):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put train.csv, store.csv and test.csv inside data/.")
    return pd.read_csv(path, **kwargs)


def load_train_store():
    train = read_csv("train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"], low_memory=False)
    return train, read_csv("store.csv")


def load_test_store():
    test = read_csv("test.csv", dtype={"StateHoliday": str}, parse_dates=["Date"], low_memory=False)
    return test, read_csv("store.csv")


def add_date_features(df):
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


def get_feature_lists(extended=False):
    numeric, categorical = list(BASE_NUMERIC), list(BASE_CATEGORICAL)
    if extended:
        numeric += EXTRA_NUMERIC
        categorical += EXTRA_CATEGORICAL
    return numeric, categorical


def prepare_training_data(extended=False):
    """Merge train with store, drop zero-sales rows, return X, y and feature lists."""
    train, store = load_train_store()
    df = pd.merge(train, store, on="Store", how="inner")
    df = df[df[TARGET] > 0]
    df = add_date_features(df)
    numeric, categorical = get_feature_lists(extended)
    return df[numeric + categorical], df[TARGET], numeric, categorical


def build_preprocessor(numeric, categorical):
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="mean"), numeric),
        ("cat", categorical_tf, categorical),
    ])


def build_xgb_model(numeric, categorical, n_estimators=100, learning_rate=0.1):
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(numeric, categorical)),
        ("regressor", XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                   random_state=RANDOM_STATE, n_jobs=-1)),
    ])


# --------------------------------------------------------------------------- #
# 1. EDA
# --------------------------------------------------------------------------- #
def clean_store(store):
    store = store.copy()
    store["CompetitionDistance"] = store["CompetitionDistance"].fillna(store["CompetitionDistance"].median())
    for col in ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "Promo2SinceWeek", "Promo2SinceYear"]:
        store[col] = store[col].fillna(0)
    store["PromoInterval"] = store["PromoInterval"].fillna("")

    store["HasCompetition"] = (store["CompetitionOpenSinceYear"] > 0).astype(int)
    store["HasPromo2"] = (store["Promo2"] == 1).astype(int)
    store["CompetitionTenure"] = np.where(
        store["HasCompetition"] == 1, REFERENCE_YEAR - store["CompetitionOpenSinceYear"], 0)
    for m in MONTHS:
        store[f"Promo_{m}"] = store["PromoInterval"].str.contains(m).astype(int)
    store["CompetitionDistanceCategory"] = pd.cut(
        store["CompetitionDistance"],
        bins=[0, 500, 1000, 2000, 5000, 10000, 20000, 50000, float("inf")],
        labels=["Very Close", "Close", "Medium", "Far", "Very Far", "Distant", "Very Distant", "Extremely Distant"])
    return store


def remove_outliers_iqr(df, column):
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]


def run_eda(args):
    import matplotlib
    matplotlib.use("Agg")  # save figures instead of opening windows
    import matplotlib.pyplot as plt
    import plotly.express as px
    import seaborn as sns

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def save(name):
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / name, dpi=150)
        plt.close()
        print(f"saved {FIGURES_DIR / name}")

    _, raw_store = load_train_store()
    print("Missing values (raw):\n", raw_store.isnull().sum())
    try:
        import missingno as msno
        msno.matrix(raw_store)
        save("missing_values.png")
    except ImportError:
        print("missingno not installed - skipping missing-values matrix")

    store = clean_store(raw_store)
    print("\nMissing values (after cleaning):", int(store.isnull().sum().sum()))

    # Distributions
    for col in ["CompetitionDistance", "CompetitionTenure"]:
        plt.figure(figsize=(8, 6))
        sns.histplot(store[col], bins=30, kde=True)
        plt.axvline(store[col].mean(), color="r", linestyle="--", label="Mean")
        plt.axvline(store[col].median(), color="g", linestyle="-", label="Median")
        plt.title(f"Distribution of {col}")
        plt.legend()
        save(f"dist_{col}.png")

    # Before / after outlier removal
    cleaned = remove_outliers_iqr(remove_outliers_iqr(store, "CompetitionDistance"), "CompetitionTenure")
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    for j, col in enumerate(["CompetitionDistance", "CompetitionTenure"]):
        sns.histplot(store[col], bins=50, kde=True, ax=axes[0, j]).set_title(f"Before cleaning - {col}")
        sns.histplot(cleaned[col], bins=50, kde=True, color="green", ax=axes[1, j]).set_title(
            f"After cleaning - {col}")
    save("outliers_before_after.png")

    # Correlation + pairplot
    numeric_features = ["CompetitionDistance", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
                        "Promo2SinceWeek", "Promo2SinceYear", "CompetitionTenure"]
    plt.figure(figsize=(12, 8))
    sns.heatmap(store[numeric_features].corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix of Numerical Features")
    save("correlation_matrix.png")

    g = sns.pairplot(store[numeric_features], diag_kind="kde")
    g.figure.suptitle("Pairplot of Numerical Features", y=1.02)
    g.savefig(FIGURES_DIR / "pairplot.png", dpi=100)
    plt.close("all")

    # Interactive plots (HTML)
    px.box(store, x="StoreType", y="CompetitionDistance", points=False, log_y=True,
           title="Competition Distance by Store Type").write_html(FIGURES_DIR / "box_distance_by_type.html")
    px.scatter(store, x="CompetitionTenure", y="CompetitionDistance", color="StoreType", log_y=True,
               title="Competition Tenure vs Distance").write_html(FIGURES_DIR / "scatter_tenure_distance.html")
    promo = store.groupby("StoreType")["Promo2"].mean().reset_index()
    px.bar(promo, x="StoreType", y="Promo2", title="Promo2 Participation Rate by Store Type",
           labels={"Promo2": "Participation Rate"}).write_html(FIGURES_DIR / "bar_promo2_by_type.html")
    print("saved interactive HTML plots")


# --------------------------------------------------------------------------- #
# 2. Train + evaluate
# --------------------------------------------------------------------------- #
def run_train(args):
    X, y, numeric, categorical = prepare_training_data(args.extended)
    print(f"Rows: {len(X):,} | Features: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = build_xgb_model(numeric, categorical, args.n_estimators, args.learning_rate)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "extended_features": args.extended,
    }
    print("\nEvaluation on test set:")
    print(f"  R2   : {metrics['r2']:.4f}")
    print(f"  MAE  : {metrics['mae']:.2f}")
    print(f"  RMSE : {metrics['rmse']:.2f}")

    if args.cv:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        metrics["cv_rmse_scores"] = [round(float(s), 2) for s in scores]
        metrics["cv_rmse_mean"] = round(float(scores.mean()), 2)
        print("\nCross-validation RMSE:", metrics["cv_rmse_scores"])
        print("Average RMSE:", metrics["cv_rmse_mean"])

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "extended": args.extended}, MODEL_PATH)
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {OUTPUTS_DIR / 'metrics.json'}")


# --------------------------------------------------------------------------- #
# 3. Hyperparameter tuning
# --------------------------------------------------------------------------- #
def run_tune(args):
    X, y, numeric, categorical = prepare_training_data(args.extended)
    if args.sample and args.sample < len(X):
        idx = X.sample(args.sample, random_state=RANDOM_STATE).index
        X, y = X.loc[idx], y.loc[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(numeric, categorical)),
        ("regressor", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_dist = {
        "regressor__n_estimators": randint(50, 200),
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": randint(2, 6),
        "regressor__min_samples_leaf": randint(1, 4),
    }
    search = RandomizedSearchCV(pipeline, param_dist, n_iter=args.n_iter, cv=3,
                                scoring="neg_mean_squared_error", n_jobs=1, verbose=2,
                                random_state=RANDOM_STATE)
    search.fit(X_train, y_train)

    rmse = float(np.sqrt(mean_squared_error(y_test, search.predict(X_test))))
    params = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in search.best_params_.items()}
    print("Best parameters:", params)
    print(f"Test RMSE with best params: {rmse:.2f}")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "best_params.json").write_text(json.dumps({"params": params, "test_rmse": rmse}, indent=2))


# --------------------------------------------------------------------------- #
# 4. Predict on test.csv
# --------------------------------------------------------------------------- #
def run_predict(args):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python rossmann.py train` first.")
    artifact = joblib.load(MODEL_PATH)
    model, features = artifact["model"], artifact["features"]

    test, store = load_test_store()
    df = add_date_features(pd.merge(test, store, on="Store", how="left"))

    preds = model.predict(df[features])
    preds[df["Open"].fillna(1).to_numpy() == 0] = 0  # closed stores -> 0 sales

    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / "submission.csv"
    pd.DataFrame({"Id": df["Id"], "Sales": preds.round(2)}).to_csv(path, index=False)
    print(f"Saved {len(df):,} predictions to {path}")


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Rossmann Store Sales Prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("eda", help="exploratory data analysis").set_defaults(func=run_eda)

    p = sub.add_parser("train", help="train and evaluate the XGBoost model")
    p.add_argument("--extended", action="store_true", help="use extra daily features")
    p.add_argument("--cv", action="store_true", help="run 5-fold cross-validation")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.set_defaults(func=run_train)

    p = sub.add_parser("tune", help="hyperparameter tuning (RandomForest)")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--sample", type=int, default=100_000, help="rows used for tuning (0 = all)")
    p.add_argument("--n-iter", type=int, default=20)
    p.set_defaults(func=run_tune)

    sub.add_parser("predict", help="predict sales for data/test.csv").set_defaults(func=run_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
