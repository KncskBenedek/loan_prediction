from zenml import step
from zenml.client import Client

@step
def register_model(model):
    client = Client()
    client.model_registry.get_default().register_model(
        model=model,
        name="xgb-loan-predictor",
        version="1.0.0"
    )