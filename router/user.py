from  fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
#from schema.user import *
from make_api.jwt import *
user_router=APIRouter

class User(BaseModel):
    email:str
    password:str


@user_router('/login',tags=[''])
def login(user:User):
    if user.email=='juan@gmail.com' and user.password=='juan123':
        token: str =create_token(user.dict())
        return HTMLResponse(status_code=200, content=token)
    else:
        return HTMLResponse(status_code=401, content={"message": "Credenciales inválidas, intente de nuevo"})