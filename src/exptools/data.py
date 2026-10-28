"""Loading the Hillstrom MineThatData e-mail campaign (64,000 customers, 3 randomized arms)."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

# The original host (minethatdata.com) is intermittently offline; this is a public mirror.
MIRROR_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "hillstrom.csv"

ARMS = {"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 2}
CATEGORICAL = ["zip_code", "channel"]
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]


def load_hillstrom(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the dataset, downloading it on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        gz = path.with_suffix(".csv.gz")
        urllib.request.urlretrieve(MIRROR_URL, gz)
        with gzip.open(gz, "rb") as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()
    df = pd.read_csv(path)
    df["arm"] = df["segment"].map(ARMS)
    df["treated"] = (df["arm"] > 0).astype(int)  # any e-mail vs no e-mail
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment features only, with categoricals one-hot encoded."""
    return pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
