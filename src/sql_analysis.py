import re
import sqlite3

import pandas as pd

from clean import ROOT, load_clean

SQL_FILE = ROOT / "sql" / "analysis.sql"
DB_FILE = ROOT / "data" / "cars.db"
OUT_DIR = ROOT / "results"


def parse_queries(text: str) -> dict[str, str]:
    return {
        name: query.strip()
        for name, query in re.findall(r"-- name: (\w+)\n(.*?)(?=\n-- name:|\Z)", text, re.S)
    }


def main() -> None:
    with sqlite3.connect(DB_FILE) as conn:
        load_clean().to_sql("cars", conn, if_exists="replace", index=False)
        for name, query in parse_queries(SQL_FILE.read_text()).items():
            result = pd.read_sql_query(query, conn)
            result.to_csv(OUT_DIR / f"sql_{name}.csv", index=False)
            print(f"\n=== {name} ===\n{result.to_string(index=False)}")


if __name__ == "__main__":
    main()
