from zenml import step
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

@step
def deploy_model(model):
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    service = deployer.deploy_model(
        model=model,
        model_name="xgb-loan-predictor",
        replace=True
    )