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