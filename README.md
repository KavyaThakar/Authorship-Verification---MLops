# Authorship Verification — MLOps Pipeline

> **From Experiment to Endpoint: Tracked Model Training and API Deployment**

This project converts an existing **Authorship Verification** machine-learning model into a reproducible MLOps pipeline.

The original model predicts whether two text samples were written by the **same author** or by **different authors**. The MLOps implementation adds experiment tracking with **MLflow**, model registration, a **FastAPI** inference service, input validation, and **Docker** containerization.

---

## 1. Project Overview

### Problem

Given two text samples:

- `text1`
- `text2`

the system predicts whether they are likely to have been written by the same author.

### MLOps Goal

Instead of only training a model in a notebook, this project creates a complete workflow:

```text
Dataset
   │
   ▼
Feature Extraction
   │
   ▼
Model Training
   │
   ├──────────────► MLflow Parameters
   ├──────────────► MLflow Metrics
   └──────────────► MLflow Model Artifact
                         │
                         ▼
                 Compare 7 Runs
                         │
                         ▼
                 Select Best Model
                         │
                         ▼
                MLflow Model Registry
                         │
                         ▼
                    Model v1
                         │
                         ▼
                    FastAPI API
                         │
                         ▼
                     Docker
                         │
                         ▼
                  /health + /predict
