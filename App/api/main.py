from fastapi import FastAPI
from .models.IrisInput import IrisInput
from .load_model import load_prod_model
import numpy as np
from sklearn import datasets
import mlflow

IRIS_CLASS_NAMES=["setosa","versicolor", "virginica"]
mlflow.set_tracking_uri("http://127.0.0.1:5000")

  
def preprocess_input(data: IrisInput) -> np.ndarray:
    return np.array([[
        data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    ]], dtype=np.float32)

app = FastAPI()
iris_model = load_prod_model("iris_model", "latest")

@app.post("/iris/predict")
async def predict_iris(iris: IrisInput):
    y_pred = IRIS_CLASS_NAMES[iris_model.predict(preprocess_input(iris))[0]]
    return y_pred 
