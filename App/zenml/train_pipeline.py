from zenml import run_pipeline
from steps.load_data import load_data
from steps.transform_data import transform_data
from steps.train_model import train_model
from step.evaluate_model import evaluate_model
def train_pipeline():
    data = load_data("./data/Loan Prediction.csv")
    X_train, X_test, y_train, y_test = transform_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    register_model(model)
    deploy_model(model)