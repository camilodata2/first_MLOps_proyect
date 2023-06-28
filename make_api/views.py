from make_api.models import *
from make_api.utils import *

model=None

def get_prediction(request:PredictionRequest) -> float:
    data_to_predict=transform_to_dataframe(request)
    prediction=model.predict(data_to_predict)[0]
    return max(0,prediction ) #nunca entregar una predicion gruda aun usuario final