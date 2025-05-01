import numpy as np
from pydantic import BaseModel

IRIS_CLASS_NAMES=["setosa","versicolor", "virginica"]


class IrisInput(BaseModel):
    petal_length:float
    petal_width:float
    sepal_length:float
    sepal_width:float
class IrisOutput(BaseModel):
    species:str
def preprocess_iris_input(data: IrisInput) -> np.ndarray:
    return np.array([[
    data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
]], dtype=np.float64)
