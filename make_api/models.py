from pydantic import BaseModel,fields

class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: float
    education : str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race:str
    sex: str
    capital_gain: None
    capital_loss : str
    hours_per_week: float
    native_contry: str
    income : float

class PredictionResponse(BaseModel):
    adult:float
    

