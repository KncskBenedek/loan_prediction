import json
from fastapi import FastAPI
from .models.Iris import IrisInput,IrisOutput, IRIS_CLASS_NAMES, preprocess_iris_input
from .load_model import load_prod_model
import numpy as np
from sklearn import datasets
import mlflow
from .models.Loan import LoanInput,LoanOutput, preprocess_loan_input
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from prometheus_fastapi_instrumentator.metrics import Info
from typing import Callable
mlflow.set_tracking_uri("http://mlflow:5000")


app = FastAPI()

def predicted_iris_class_name() -> Callable[[Info], None]:
    METRIC = Counter(
        "predicted_iris_class_name",
        "Number of times a certain class predicted",
        labelnames=("species",)
    )

    def instrumentation(info: Info) -> None:
        if info.request.url.path == '/iris/predict' and info.response.status_code == 200:
            class_name = json.loads(info.response.body)['species']
            METRIC.labels(class_name).inc()
    return instrumentation

def predicted_loan_risk_flag() -> Callable[[Info], None]:
    METRIC = Counter(
        "predicted_loan_risk_flag",
        "Number of times a certain class predicted",
        labelnames=("risk_flag",)
    )

    def instrumentation(info: Info) -> None:
        if info.request.url.path == '/loan/predict' and info.response.status_code == 200:
            class_name = json.loads(info.response.body)['risk_flag']
            METRIC.labels(class_name).inc()

    return instrumentation
Instrumentator().instrument(app).expose(app)
(
    Instrumentator(body_handlers=[r".*"])
        .instrument(app)
        .add(predicted_iris_class_name(), predicted_loan_risk_flag() )
        .expose(app)
    )

iris_model = load_prod_model("iris_model")
loan_model=load_prod_model("loan_model")
@app.post("/iris/predict")
async def predict_iris(iris: IrisInput)->IrisOutput:
    y_pred = IRIS_CLASS_NAMES[iris_model.predict(preprocess_iris_input(iris))[0]]
    return {"species":y_pred}

@app.post("/loan/predict")
async def predict_loan(loan: LoanInput)->LoanOutput:
    y_pred = str(loan_model.predict(preprocess_loan_input(loan))[0])
    return {"risk_flag":y_pred}

