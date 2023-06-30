from starlette.middleware.base import  BaseHTTPMiddleware, RequestResponseEndpoint
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response
from fastapi.responses import JSONResponse

class Manejo_de_erro(BaseHTTPMiddleware):
    def __init__(self,app: FastAPI) -> None:
        super.__init__(app)

    
    async def dispatch(self, request: Request, call_next) -> Response | JSONResponse:
        try:
            return await call_next(request)
        except Exception as e:
            return JSONResponse(status_code=500, content={'error': str(e)})