from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from make_api.views import *
from fastapi.responses import HTMLResponse
from make_api.jwt import *
from make_api.models import PredictionRequest
from middleware.error import *
app=FastAPI(docs_url='/')
app.title='ML model'
app.version='0.0.1'

app.middleware(Manejo_de_erro)
#app.include_router(user_router)

@app.get("/")
def root():
    return {"HELLO":"WORD"}

class Client(BaseModel):
    email: str
    password: str


@app.post('/login',tags=['autenticacion'])
def loging_user(user:Client):
    if user.email == "juan@gmail.com" and user.password == "juan123" :
        token: str =create_token(user.dict())
        return HTMLResponse(status_code=200, content=token)
    else:
        return HTMLResponse(status_code=401, content={"message": "Credenciales inválidas, intente de nuevo"})

@app.post('/v1/prediction',tags=['my first prediction'])
def make_model_prediction(request:PredictionRequest):
    return PredictionResponse(adult=get_prediction(request))