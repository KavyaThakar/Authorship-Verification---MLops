import mlflow
import mlflow.xgboost
import os
import shutil

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_uri = "models:/AuthorshipVerificationModel/1"

output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model"
)

print("Loading registered model...")
model = mlflow.xgboost.load_model(model_uri)

print("Saving model for Docker...")

mlflow.xgboost.save_model(
    model,
    output_path
)

print("Model exported successfully.")
print("Model location:", output_path)