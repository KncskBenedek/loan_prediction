from fastapi import FastAPI
from .models.Iris import IrisInput, IRIS_CLASS_NAMES, preprocess_iris_input
from .load_model import load_prod_model
import numpy as np
from sklearn import datasets
import mlflow
from .models.Loan import LoanInput, preprocess_loan_input


mlflow.set_tracking_uri("http://127.0.0.1:5000")



app = FastAPI()
iris_model = load_prod_model("iris_model")
loan_model=load_prod_model("loan_model")

@app.post("/iris/predict")
async def predict_iris(iris: IrisInput):
    y_pred = IRIS_CLASS_NAMES[iris_model.predict(preprocess_iris_input(iris))[0]]
    return y_pred 

@app.post("/loan/predict")
async def predict_loan(loan: LoanInput):
    y_pred = str(loan_model.predict(preprocess_loan_input(loan))[0])
    return y_pred