import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=["1"],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1_score DESC"]
)

best = runs[0]

print("BEST RUN:", best.info.run_id)
print("BEST F1:", best.data.metrics["f1_score"])

model_uri = f"runs:/{best.info.run_id}/authorship_xgboost_model"

print("MODEL URI:", model_uri)

result = mlflow.register_model(
    model_uri,
    "AuthorshipVerificationModel"
)

print("REGISTERED MODEL:", result.name)
print("MODEL VERSION:", result.version)