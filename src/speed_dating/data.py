"""Loading and cleaning of the Speed Dating dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "speed_dating.csv"
RESULTS_DIR = REPO_ROOT / "results"

# The six attributes participants rate; 'shar' (shared interests) is missing
# from the "how do you rate yourself" questions (3_x and 5_x).
ATTRIBUTES = ["attr", "sinc", "intel", "fun", "amb", "shar"]
ATTRIBUTE_NAMES = {
    "attr": "Attractive",
    "sinc": "Sincere",
    "intel": "Intelligent",
    "fun": "Fun",
    "amb": "Ambitious",
    "shar": "Shared interests",
}

GENDER = {0: "Female", 1: "Male"}

RACE = {
    1: "Black/African American",
    2: "European/Caucasian-American",
    3: "Latino/Hispanic American",
    4: "Asian/Pacific Islander/Asian-American",
    5: "Native American",
    6: "Other",
}

CAREER = {
    1: "Lawyer",
    2: "Academic/Research",
    3: "Psychologist",
    4: "Doctor/Medicine",
    5: "Engineer",
    6: "Creative Arts/Entertainment",
    7: "Business/Finance/Consulting",
    8: "Real Estate",
    9: "International/Humanitarian Affairs",
    10: "Undecided",
    11: "Social Work",
    12: "Speech Pathology",
    13: "Politics",
    14: "Pro sports/Athletics",
    15: "Other",
    16: "Journalism",
    17: "Architecture",
}

GOAL = {
    1: "Fun night out",
    2: "Meet new people",
    3: "Get a date",
    4: "Serious relationship",
    5: "To say I did it",
    6: "Other",
}

FREQUENCY = {
    1: "Several times a week",
    2: "Twice a week",
    3: "Once a week",
    4: "Twice a month",
    5: "Once a month",
    6: "Several times a year",
    7: "Almost never",
}


def load_data(path=DATA_PATH):
    """Read the raw CSV (one row per date, seen from one participant)."""
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df.drop_duplicates().reset_index(drop=True)


def attribute_columns(suffix, attributes=ATTRIBUTES):
    """attribute_columns('1_1') -> ['attr1_1', 'sinc1_1', ...]."""
    return [f"{a}{suffix}" for a in attributes]


def participants(df):
    """One row per participant: the questionnaire answers repeat on every date row."""
    return df.drop_duplicates("iid").set_index("iid")


def scale_check(df, columns, threshold=60):
    """Classify each row of a 100-point question by the sum of its answers.

    Most waves split 100 points between the attributes, but some waves rated
    every attribute on a 1-10 scale instead (sum <= 60), and some rows sum to
    a value close to, but not exactly, 100.
    """
    total = df[columns].sum(axis=1, min_count=len(columns))
    kind = np.select(
        [total.isna(), total <= threshold, total.round(1) == 100],
        ["missing", "1-10 scale", "100 points"],
        default="100 points, wrong sum",
    )
    return pd.DataFrame({"sum": total, "kind": kind}, index=df.index)


def normalize_to_100(df, columns, threshold=60):
    """Bring every answer of a 100-point question to shares that sum to 100.

    Rows answered on the 1-10 scale have their zeros raised to 1 (the lowest
    valid rating) before rescaling; rows with a slightly wrong sum are simply
    rescaled. Rows with missing answers are left as they are.
    """
    out = df.copy()
    values = out[columns].astype(float)
    total = values.sum(axis=1, min_count=len(columns))
    ten_scale = total <= threshold
    values.loc[ten_scale] = values.loc[ten_scale].clip(lower=1)
    total = values.sum(axis=1, min_count=len(columns))
    out[columns] = values.mul(100.0 / total, axis=0)
    return out


# Waves with a special set-up (from the dataset description).
SPECIAL_WAVES = {
    "rated_1_10": [6, 7, 8, 9],     # preferences rated 1-10 instead of 100 points
    "on_a_budget": [12],            # could say "yes" to at most 50% of partners
    "with_magazine": [18, 21],      # asked to bring a magazine
    "with_book": [19, 20],          # asked to bring a book
}


def add_special_wave_flags(df):
    out = df.copy()
    for name, waves in SPECIAL_WAVES.items():
        out[name] = out["wave"].isin(waves).astype(int)
    return out
