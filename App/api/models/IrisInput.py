from pydantic import BaseModel

class IrisInput(BaseModel):
    petal_length:float
    petal_width:float
    sepal_length:float
    sepal_width:float