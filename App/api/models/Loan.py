import pandas as pd
from pydantic import BaseModel


class LoanInput(BaseModel):
    income:int
    age:int
    experience:int
    marital_status:str
    house_ownership:str
    car_ownership:str
    profession:str
    city:str
    state:str
    current_job_years:int
    current_house_years:int

def preprocess_loan_input(data: LoanInput) -> pd.DataFrame:
    return pd.DataFrame({
    'Income':[data.income],
    'Age':[data.age], 
    'Experience':[data.experience], 
    'Married/Single': [data.marital_status], 
    'House_Ownership':[data.house_ownership],
    'Car_Ownership'    :[data.car_ownership],
    'Profession'    :[data.profession],
    'CITY'    :[data.city],
    'STATE'    :[data.state],
    'CURRENT_JOB_YRS'    :[data.current_job_years],
    'CURRENT_HOUSE_YRS'    :[data.current_house_years]
    })