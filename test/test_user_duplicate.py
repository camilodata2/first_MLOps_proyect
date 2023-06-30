from fastapi.testclient import TestClient
from main import app

cliente=TestClient(app)

def test_create_user_duplicate_email():
    user={
        'email':'test_create_user_duplicatejuan@gmail.com',
        'username':'test_create_user_duplicate',
        'password':'test_create_user_duplicateadmin123'
    }
    response=cliente.post('api/v1/user',json=user)

    assert response.status_code==201 , response.txt
    
    user['email']='test_create_user_duplicate_email2'
    
    response=cliente.post('api/v1/user',json=user)
    
    assert response.status_code==400 ,response.txt
    
    data=response.json()

    assert data['detail']=='Email already registered'

def test_create_user_duplicate_username():
  

    user = {
        'email': 'test_create_user_duplicate_username@cosasdedevs.com',
        'username': 'test_create_user_duplicate_username',
        'password': 'admin123'
    }

    response = cliente.post(
        '/api/v1/user/',
        json=user,
    )
    assert response.status_code == 201, response.text

    response = cliente.post(
        '/api/v1/user/',
        json=user,
    )
    assert response.status_code == 400, response.text
    data = response.json()
    assert data['detail'] == 'Username already registered'