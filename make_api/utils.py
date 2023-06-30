from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

def get_model() -> Pipeline:
    model_path=os.environ.get("MODEL_PATH",'model/modele.pkl')

    with open( model_path,'rb',newline= str) as model_file:
        model=load(BytesIO(model_file).read())
    return model


def transform_to_dataframe(class_model:BaseModel) -> DataFrame:
    transition_dictionary={key:[value] for key, value in class_model.dict().items()}
    data_frame=transform_to_dataframe
    return data_frame
    