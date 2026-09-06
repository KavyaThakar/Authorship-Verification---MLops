import os
import mlflow
import mlflow.xgboost
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_NAMES_FILE = os.path.join(
    BASE_DIR,
    "mlops",
    "feature_names.txt"
)


# ============================================================
# MLflow CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")


MODEL_NAME = "AuthorshipVerificationModel"
MODEL_VERSION = "1"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model"
)


print("Loading authorship verification model...")

model = mlflow.xgboost.load_model(MODEL_PATH)

print("Model loaded successfully.")



# ============================================================
# LOAD FEATURE ORDER
# ============================================================

with open(
    FEATURE_NAMES_FILE,
    "r",
    encoding="utf-8"
) as f:
    FEATURE_NAMES = [
        line.strip()
        for line in f
        if line.strip()
    ]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Authorship Verification API",
    description="API for verifying whether two texts were written by the same author.",
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    text1: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="First text"
    )

    text2: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Second text"
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):

    words = text.split()

    sentences = [
        s.strip()
        for s in text.replace(
            "!",
            "."
        ).replace(
            "?",
            "."
        ).split(".")
        if s.strip()
    ]

    word_lengths = [
        len(w)
        for w in words
    ]

    avg_word_len = (
        np.mean(word_lengths)
        if word_lengths
        else 0.0
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

    comma_freq = (
        text.count(",")
        / max(len(words), 1)
    )

    semicolon_freq = (
        text.count(";")
        / max(len(words), 1)
    )

    punctuation_count = sum(
        text.count(c)
        for c in [
            ".",
            ",",
            ";",
            ":",
            "!",
            "?"
        ]
    )

    punctuation_ratio = (
        punctuation_count
        / max(len(text), 1)
    )

    uppercase_ratio = (
        sum(
            c.isupper()
            for c in text
        )
        /
        max(
            sum(
                c.isalpha()
                for c in text
            ),
            1
        )
    )

    digit_ratio = (
        sum(
            c.isdigit()
            for c in text
        )
        / max(len(text), 1)
    )

    # Readability approximation
    sentence_count = max(
        len(sentences),
        1
    )

    word_count = max(
        len(words),
        1
    )

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

        syllable_count += max(
            syllables,
            1
        )

    flesch_reading_ease = (
        206.835
        - 1.015
        * (
            word_count
            / sentence_count
        )
        - 84.6
        * (
            syllable_count
            / word_count
        )
    )

    gunning_fog = (
        0.4
        * (
            (
                word_count
                / sentence_count
            )
            +
            100
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
# CREATE MODEL INPUT
# ============================================================

def create_prediction_features(text1, text2):

    features1 = extract_features(text1)
    features2 = extract_features(text2)

    differences = {
        key: abs(
            features1[key]
            - features2[key]
        )
        for key in features1
    }

    return np.array(
        [
            differences[name]
            for name in FEATURE_NAMES
        ]
    ).reshape(1, -1)


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        features = create_prediction_features(
            request.text1,
            request.text2
        )

        prediction = int(
            model.predict(features)[0]
        )

        probability = float(
            model.predict_proba(features)[0][1]
        )

        if prediction == 1:
            result = "Same Author"
        else:
            result = "Different Author"

        return {
            "prediction": result,
            "probability": round(
                probability,
                4
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )