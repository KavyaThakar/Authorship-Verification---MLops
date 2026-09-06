import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

PAIRS_FILE = os.path.join(
    DATA_DIR,
    "pan20-authorship-verification-training-small.jsonl"
)

TRUTH_FILE = os.path.join(
    DATA_DIR,
    "pan20-authorship-verification-training-small-truth.jsonl"
)

FEATURE_CACHE = os.path.join(
    BASE_DIR,
    "mlops",
    "features_cache.csv"
)

FEATURE_NAMES_FILE = os.path.join(
    BASE_DIR,
    "mlops",
    "feature_names.txt"
)

# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Authorship_Verification")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):
    words = text.split()

    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]

    word_lengths = [len(w) for w in words]

    avg_word_len = (
        np.mean(word_lengths) if word_lengths else 0.0
    )

    type_token_ratio = (
        len(set(words)) / len(words)
        if words
        else 0.0
    )

    avg_sentence_len = (
        len(words) / len(sentences)
        if sentences
        else 0.0
    )

    comma_freq = text.count(",") / max(len(words), 1)

    semicolon_freq = text.count(";") / max(len(words), 1)

    punctuation_count = sum(
        text.count(c)
        for c in [".", ",", ";", ":", "!", "?"]
    )

    punctuation_ratio = (
        punctuation_count / max(len(text), 1)
    )

    uppercase_ratio = (
        sum(c.isupper() for c in text)
        / max(sum(c.isalpha() for c in text), 1)
    )

    digit_ratio = (
        sum(c.isdigit() for c in text)
        / max(len(text), 1)
    )

    # Simple readability approximations
    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)

    syllable_count = 0

    for word in words:
        word = word.lower()
        vowels = "aeiouy"

        syllables = 0
        previous_vowel = False

        for char in word:
            is_vowel = char in vowels

            if is_vowel and not previous_vowel:
                syllables += 1

            previous_vowel = is_vowel

        if word.endswith("e") and syllables > 1:
            syllables -= 1

        syllable_count += max(syllables, 1)

    flesch_reading_ease = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllable_count / word_count)
    )

    gunning_fog = (
        0.4
        * (
            (word_count / sentence_count)
            + 100
            * (
                sum(
                    1
                    for word in words
                    if len(word) > 6
                )
                / word_count
            )
        )
    )

    return {
        "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "avg_sentence_len": avg_sentence_len,
        "comma_freq": comma_freq,
        "semicolon_freq": semicolon_freq,
        "punctuation_ratio": punctuation_ratio,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "flesch_reading_ease": flesch_reading_ease,
        "gunning_fog": gunning_fog,
    }


# ============================================================
# LOAD TRUTH LABELS
# ============================================================

def load_truth():
    truth = {}

    print("Loading truth labels...")

    with open(TRUTH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            truth[record["id"]] = int(record["same"])

    print(f"Truth records loaded: {len(truth)}")

    return truth


# ============================================================
# CREATE / LOAD FEATURE CACHE
# ============================================================

def create_feature_cache():

    if os.path.exists(FEATURE_CACHE):
        print("\nFeature cache already exists.")
        print("Loading cached features...")
        return pd.read_csv(FEATURE_CACHE)

    print("\nFeature cache not found.")
    print("Extracting features from dataset...")
    print("This may take some time ONLY the first time.\n")

    truth = load_truth()

    rows = []

    with open(PAIRS_FILE, "r", encoding="utf-8") as f:

        for count, line in enumerate(f, start=1):

            pair_data = json.loads(line)

            pair_id = pair_data["id"]

            if pair_id not in truth:
                continue

            texts = pair_data["pair"]

            text1 = texts[0]
            text2 = texts[1]

            features1 = extract_features(text1)
            features2 = extract_features(text2)

            row = {
                key: abs(features1[key] - features2[key])
                for key in features1
            }

            row["label"] = truth[pair_id]

            rows.append(row)

            if count % 5000 == 0:
                print(
                    f"Processed {count} pairs..."
                )

    df = pd.DataFrame(rows)

    df.to_csv(
        FEATURE_CACHE,
        index=False
    )

    feature_names = [
        c for c in df.columns
        if c != "label"
    ]

    with open(
        FEATURE_NAMES_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        for name in feature_names:
            f.write(name + "\n")

    print("\nFeature extraction completed.")
    print(f"Total usable pairs: {len(df)}")
    print(f"Cached at: {FEATURE_CACHE}")

    return df


# ============================================================
# LOAD FEATURES
# ============================================================

df = create_feature_cache()

FEATURE_NAMES = [
    c for c in df.columns
    if c != "label"
]

X = df[FEATURE_NAMES]
y = df["label"]

print("\nFeatures:")
print(FEATURE_NAMES)

print("\nDataset shape:")
print(X.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6 MEANINGFUL XGBOOST CONFIGURATIONS
# ============================================================

RUN_CONFIGS = [

    # Run 1 - baseline
    {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # Run 2 - shallower trees
    {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # Run 3 - deeper trees
    {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # Run 4 - slower learning rate
    {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },

    # Run 5 - more subsampling
    {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },

    # Run 6 - larger model
    {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
]


# ============================================================
# TRAIN MULTIPLE MLflow RUNS
# ============================================================

results = []

print("\n")
print("=" * 60)
print("STARTING MLflow EXPERIMENTS")
print("=" * 60)


for run_number, params in enumerate(
    RUN_CONFIGS,
    start=1
):

    print("\n")
    print("-" * 60)
    print(f"RUN {run_number} / {len(RUN_CONFIGS)}")
    print("-" * 60)

    with mlflow.start_run(
        run_name=f"xgboost_run_{run_number}"
    ):

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_params(params)

        mlflow.log_param(
            "model",
            "XGBoost"
        )

        mlflow.log_param(
            "feature_count",
            len(FEATURE_NAMES)
        )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric="logloss",
            random_state=42,
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
        }

        mlflow.log_metrics(metrics)

        # ----------------------------------------------------
        # Log model
        # ----------------------------------------------------

        mlflow.xgboost.log_model(
            model,
            "authorship_xgboost_model"
        )

        # ----------------------------------------------------
        # Log feature names
        # ----------------------------------------------------

        mlflow.log_artifact(
            FEATURE_NAMES_FILE
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Accuracy    : {accuracy:.4f}"
        )

        print(
            f"Precision   : {precision:.4f}"
        )

        print(
            f"Recall      : {recall:.4f}"
        )

        print(
            f"Specificity : {specificity:.4f}"
        )

        print(
            f"F1 Score    : {f1:.4f}"
        )

        results.append({
            **params,
            **metrics
        })


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MLflow EXPERIMENT RESULTS")
print("=" * 60)

print(
    results_df[
        [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1_score",
        ]
    ].to_string(index=False)
)

best_index = results_df[
    "f1_score"
].idxmax()

best_result = results_df.loc[
    best_index
]

print("\n")
print("=" * 60)
print("BEST CONFIGURATION")
print("=" * 60)

print(
    best_result.to_string()
)

print("\nAll MLflow runs completed successfully.")