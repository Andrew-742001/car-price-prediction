import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import ROOT, load_clean

IMG = ROOT / "images"
RES = ROOT / "results"
SEED = 42

NUMERIC = [
    "wheelbase", "carlength", "carwidth", "curbweight", "enginesize", "boreratio",
    "horsepower", "peakrpm", "avg_mpg", "power_to_weight", "cylindernumber",
]
CATEGORICAL = ["brand", "fueltype", "aspiration", "carbody", "drivewheel", "enginetype"]

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def build(model, scale: bool = True) -> TransformedTargetRegressor:
    prep = ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", model)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )


MODELS = {
    "Linear Regression": build(LinearRegression()),
    "Ridge Regression": build(Ridge(alpha=1.0)),
    "Random Forest": build(RandomForestRegressor(n_estimators=400, random_state=SEED), scale=False),
    "Gradient Boosting": build(GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=SEED), scale=False),
}


def eda_charts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["price"], bins=30, color="#7c5cff", edgecolor="white")
    ax.set(title="Car Price Distribution", xlabel="Price ($)", ylabel="Number of cars")
    fig.tight_layout()
    fig.savefig(IMG / "price_distribution.png")

    corr = df[NUMERIC + ["price"]].corr()["price"].drop("price").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(corr.index, corr.values, color=np.where(corr.values > 0, "#7c5cff", "#ff6b6b"))
    ax.set(title="Correlation of Each Feature with Price", xlabel="Pearson correlation")
    fig.tight_layout()
    fig.savefig(IMG / "feature_correlation.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fuel, group in df.groupby("fueltype"):
        ax.scatter(group["horsepower"], group["price"], label=fuel, alpha=0.7)
    ax.set(title="Horsepower vs Price", xlabel="Horsepower", ylabel="Price ($)")
    ax.legend(title="Fuel")
    fig.tight_layout()
    fig.savefig(IMG / "horsepower_vs_price.png")
    plt.close("all")


def evaluate(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict]:
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in MODELS.items():
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        pred = model.fit(X_train, y_train).predict(X_test)
        fitted[name] = (model, pred)
        rows.append({
            "model": name,
            "cv_r2_mean": round(cv_r2.mean(), 3),
            "cv_r2_std": round(cv_r2.std(), 3),
            "test_r2": round(r2_score(y_test, pred), 3),
            "test_mae": round(mean_absolute_error(y_test, pred), 0),
            "test_rmse": round(mean_squared_error(y_test, pred) ** 0.5, 0),
        })
    return pd.DataFrame(rows).sort_values("test_r2", ascending=False).reset_index(drop=True), fitted


def result_charts(scores: pd.DataFrame, fitted: dict, y_test: pd.Series) -> None:
    best = scores.loc[0, "model"]
    model, pred = fitted[best]

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [0, max(y_test.max(), pred.max()) * 1.05]
    ax.scatter(y_test, pred, alpha=0.75, color="#7c5cff")
    ax.plot(lim, lim, "--", color="gray", label="Perfect prediction")
    ax.set(xlim=lim, ylim=lim, title=f"{best}: Predicted vs Actual", xlabel="Actual price ($)", ylabel="Predicted price ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / "predicted_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(scores["model"][::-1], scores["test_r2"][::-1], color="#7c5cff")
    ax.set(title="Model Comparison (Test R²)", xlabel="R²", xlim=(0, 1))
    for i, v in enumerate(scores["test_r2"][::-1]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(IMG / "model_comparison.png")

    rf = fitted["Random Forest"][0].regressor_
    importance = (
        pd.Series(rf.named_steps["model"].feature_importances_, index=rf.named_steps["prep"].get_feature_names_out())
        .rename(lambda s: s.split("__", 1)[1])
        .nlargest(12)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.values, color="#7c5cff")
    ax.set(title="Top 12 Features (Random Forest Importance)", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(IMG / "feature_importance.png")
    plt.close("all")


def main() -> None:
    df = load_clean()
    eda_charts(df)

    X, y = df[NUMERIC + CATEGORICAL], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scores, fitted = evaluate(X_train, X_test, y_train, y_test)
    result_charts(scores, fitted, y_test)

    scores.to_csv(RES / "model_scores.csv", index=False)
    (RES / "summary.json").write_text(json.dumps({
        "rows": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": scores.loc[0, "model"],
        "scores": scores.to_dict(orient="records"),
    }, indent=2))
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
