"""Feature sets for predicting a speed-dating decision.

Every row of the dataset is one date seen from one participant: `dec` is that
participant's decision ("would you like to see this partner again?") and
`dec_o` is the partner's decision about the participant.
"""

from .data import attribute_columns, normalize_to_100

# Answers from the sign-up questionnaire, filled in before the event.
ACTIVITIES = [
    "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
    "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
    "music", "shopping", "yoga",
]
SELF_ATTRIBUTES = ["attr", "sinc", "intel", "fun", "amb"]
PROFILE_NUMERIC = (
    ["gender", "age", "imprace", "imprelig", "date", "go_out"]
    + ACTIVITIES
    + attribute_columns("1_1")                     # what I look for (100 points)
    + attribute_columns("3_1", SELF_ATTRIBUTES)    # how I rate myself (1-10)
)
PROFILE_CATEGORICAL = ["race", "field_cd", "goal", "career_c"]
PROFILE = PROFILE_NUMERIC + PROFILE_CATEGORICAL

# Scorecards filled in during the event, right after each 4-minute date.
MY_RATINGS = ["attr", "sinc", "intel", "fun", "amb", "shar", "like", "prob", "met"]
PARTNER_RATINGS = [f"{c}_o" for c in MY_RATINGS]

FEATURE_SETS = {
    # Model 1 of the course project: only what I said about myself at sign-up.
    "own_profile": "own sign-up profile",
    # Both sign-up profiles: everything that is known before the date.
    "both_profiles": "both sign-up profiles (before the date)",
    # Model 2/3 of the course project: profiles + both scorecards of the date.
    "profiles_and_ratings": "both profiles + both scorecards (after the date)",
}


def _partner_profiles(df):
    profiles = df.drop_duplicates("iid").set_index("iid")[PROFILE]
    return profiles.add_prefix("p_")


def build_dataset(df, target="dec", feature_set="profiles_and_ratings"):
    """Return X, y and the wave of every row (used as the CV group).

    Rows with any missing feature are dropped.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")

    data = normalize_to_100(df, attribute_columns("1_1"))
    columns = list(PROFILE)
    if feature_set != "own_profile":
        data = data.join(_partner_profiles(data), on="pid")
        columns += ["p_" + c for c in PROFILE]
    if feature_set == "profiles_and_ratings":
        columns += MY_RATINGS + PARTNER_RATINGS

    data = data.dropna(subset=columns + [target])
    return data[columns], data[target].astype(int), data["wave"]


def categorical_columns(X):
    return [c for c in X.columns
            if c in PROFILE_CATEGORICAL or c.removeprefix("p_") in PROFILE_CATEGORICAL]
