from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "CarPrice.csv"
CLEAN = ROOT / "data" / "cars_clean.csv"

BRAND_FIXES = {
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
    "alfa-romero": "alfa-romeo",
}

WORD_TO_NUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}


def load_clean() -> pd.DataFrame:
    return (
        pd.read_csv(RAW)
        .assign(
            brand=lambda d: d["CarName"].str.split().str[0].str.lower().replace(BRAND_FIXES),
            doornumber=lambda d: d["doornumber"].map(WORD_TO_NUM),
            cylindernumber=lambda d: d["cylindernumber"].map(WORD_TO_NUM),
            avg_mpg=lambda d: (d["citympg"] + d["highwaympg"]) / 2,
            power_to_weight=lambda d: d["horsepower"] / d["curbweight"] * 1000,
        )
        .drop(columns=["car_ID", "CarName"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df = load_clean()
    df.to_csv(CLEAN, index=False)
    print(f"Saved {len(df)} rows, {df['brand'].nunique()} brands -> {CLEAN.name}")
