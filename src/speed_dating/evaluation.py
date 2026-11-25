"""Leave-one-wave-out evaluation of several classifiers."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .features import categorical_columns

METRICS = {"accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC AUC"}


def make_models(random_state=0):
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000),
        "Gaussian NB": GaussianNB(),
        "SVM (linear)": SVC(kernel="linear", C=0.1),
        "KNN (k=89)": KNeighborsClassifier(n_neighbors=89),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_split=10,
            random_state=random_state, n_jobs=-1),
    }


def make_pipeline_for(X, model):
    """Encoding and scaling are fitted inside every CV fold, on the training waves only."""
    categorical = categorical_columns(X)
    numeric = [c for c in X.columns if c not in categorical]
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler(), numeric),
    ])
    return make_pipeline(preprocess, model)


def evaluate(X, y, waves, models=None):
    """Train on 20 waves, test on the remaining one, for every wave and model.

    Returns a long table: model, wave, split (train/test), metric, score.
    """
    models = models or make_models()
    splitter = LeaveOneGroupOut()
    test_waves = [waves.iloc[test].iloc[0] for _, test in splitter.split(X, y, waves)]
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            make_pipeline_for(X, model), X, y, groups=waves, cv=splitter,
            scoring=list(METRICS), return_train_score=True, n_jobs=-1)
        for split in ("train", "test"):
            for metric in METRICS:
                for wave, score in zip(test_waves, scores[f"{split}_{metric}"]):
                    rows.append((name, wave, split, metric, score))
    return pd.DataFrame(rows, columns=["model", "wave", "split", "metric", "score"])


def summary(results):
    """Mean and standard deviation over the 21 test waves."""
    table = (results.groupby(["model", "split", "metric"], sort=False)["score"]
             .agg(["mean", "std"]).unstack(["split", "metric"]))
    table = table.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1, level=[0, 1])
    return table


def test_summary(results):
    """Test scores only, formatted as 'mean ± std'."""
    test = results[results.split == "test"]
    agg = test.groupby(["model", "metric"], sort=False)["score"].agg(["mean", "std"])
    text = agg.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1).unstack("metric")
    return text[list(METRICS)].rename(columns=METRICS)


def plot_per_wave(results, metric="accuracy", title=""):
    test = results[(results.split == "test") & (results.metric == metric)]
    table = test.pivot(index="wave", columns="model", values="score")[test.model.unique()]
    ax = table.plot(figsize=(12, 5), marker="o", title=title)
    ax.set_xlabel("Test wave")
    ax.set_ylabel(f"Test {METRICS[metric].lower()}")
    ax.set_xticks(table.index)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    return ax


def plot_train_test(results, metric="accuracy", title=""):
    data = results[results.metric == metric]
    table = data.groupby(["model", "split"], sort=False)["score"].mean().unstack("split")
    ax = table[["train", "test"]].plot.bar(figsize=(10, 4), rot=0, title=title)
    ax.set_ylabel(f"Mean {METRICS[metric].lower()}")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax
