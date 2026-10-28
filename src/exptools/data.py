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


# ----------------------------------------------------------------------------- Criteo uplift v2.1

# Criteo AI Lab "Uplift Modeling" dataset (Diemert et al., 2018): 13.98M users, 12 anonymised features,
# 85% randomly assigned to an ad campaign. The official host (go.criteo.net) is not always reachable,
# so a public GitHub release mirror is used.
CRITEO_URL = "https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/criteo-research-uplift-v2.1.csv.gz"
CRITEO_DIR = Path(__file__).resolve().parents[2] / "data" / "criteo"
CRITEO_FEATURES = [f"f{i}" for i in range(12)]
CRITEO_LABELS = ["treatment", "conversion", "visit", "exposure"]


def load_criteo(directory: str | Path = CRITEO_DIR) -> pd.DataFrame:
    """Load Criteo uplift v2.1 (~14M rows). Downloads ~300 MB once, then caches a float32 array (~900 MB).

    ``exposure`` is *post-treatment* (only treated users can be exposed) so it must not be used as a feature.
    """
    import numpy as np

    directory = Path(directory)
    cols = CRITEO_FEATURES + CRITEO_LABELS
    cache = directory / "criteo_v2_1.npy"
    if not cache.exists():
        directory.mkdir(parents=True, exist_ok=True)
        gz = directory / "criteo-research-uplift-v2.1.csv.gz"
        if not gz.exists():
            urllib.request.urlretrieve(CRITEO_URL, gz)
        raw = pd.read_csv(gz, dtype={c: np.float32 for c in cols})
        np.save(cache, raw[cols].to_numpy(dtype=np.float32))
    arr = np.load(cache, mmap_mode="r")
    df = pd.DataFrame(np.asarray(arr[:, : len(CRITEO_FEATURES)]), columns=CRITEO_FEATURES)
    for i, c in enumerate(CRITEO_LABELS):
        df[c] = np.asarray(arr[:, len(CRITEO_FEATURES) + i], dtype=np.int8)
    return df
