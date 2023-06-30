from fastapi.testclient import TestClient
from main import app

client=TestClient(app)

def test_create_user_ok():
    user={
        " username":"test_create_user_ok",
        "email":"test_create_user_ok@gmail.com",
        "password":"juan123"

    }
    response=client.post('/v1/user',json=user)
    assert response.status_code==201
    data=response.json()
    assert data['email']==user['email']
    assert data['username']==user[' username']
