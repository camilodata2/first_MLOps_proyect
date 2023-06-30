from fastapi.testclient import TestClient
from main import app
from fastapi.responses import HTMLResponse
client = TestClient(app)

def test_null_prediction():
    response=client.post('/v1/prediction', json={
  "age": 0,
  "workclass":0,
  "fnlwgt": 0,
  "education": 0,
  "education_num": 0,
  "marital_status": 0,
  "occupation": 0,
  "relationship": 0,
  "race": "string",
  "sex": "string",
  "capital_gain": 0,
  "capital_loss":0,
  "hours_per_week": 0,
  "native_contry": 0,
  "income": 0
})
    if not response:
        raise HTMLResponse(status_code=404,content={"Message":"something wrong happend"})
    else:
        response.json()["income"]==0

def test_random_prediction():
    response=client.post('/vi/prediction',json={
  "age": 28,
  "workclass": 2,
  "fnlwgt": 0,
  "education": 0,
  "education_num":  5 ,
  "marital_status":0,
  "occupation": 0,
  "relationship": 0,
  "race": 'black',
  "sex": 1 ,
  "capital_gain": 0,
  "capital_loss":0,
  "hours_per_week": 0,
  "native_contry": 0,
  "income": 0
})
    assert response.status_code==201
    assert response.json()['income'] !=0
    
        
    