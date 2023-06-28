from jwt import encode,decode

def create_token(data:dict):
    token : str=encode(payload=data,key="mi_llave_secreta",algorithm="HS256")
    return token

async def validar_token(token:str) -> dict:
    data : str=decode(token,key="mi_llave_secreta",algorithms=['HS256'])
    return data