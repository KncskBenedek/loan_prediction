from zenml import step
import pandas as pd
@step
def load_data(path:str):
   return pd.read_csv(path)