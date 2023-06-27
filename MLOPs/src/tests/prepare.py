from dvc import api
import pandas as pd
import numpy as np
from io import StringIO
import sys
import logging
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('LOAGIND DATA..')
df_path=api.read('data/adult.csv', remote='dataset-tracker')

data_prepare=pd.read_csv(StringIO(df_path))

logging.info('Cleaning-dataset')
data_prepare.isnull().sum()
round((data_prepare.isin(['?']).sum() / data_prepare.shape[0]) * 100, 2).astype(str) + ' %'
df=data_prepare.replace("?",np.nan)
round((df.isnull().sum() / df.shape[0]) * 100, 2).astype(str) + ' %'
income =df['income'].value_counts(normalize=True)
round(income * 100, 2).astype('str') + ' %'
def fill_data(df,columns_with_name):
   logging.info('fill some columns')
   columns_with_nan= ['workclass', 'occupation', 'native-country']
   for col in columns_with_nan:
     df[col].fillna(df[col].mode()[0], inplace = True)
     if not columns_with_nan:
       raise ValueError('no se lleno con el metodo utilizado usa otro')
     else:
       print('vuelva a ejectutar el programa')
       return df.fillna(columns_with_name)
logging.info('Encoder category variable')
for col in df.columns:
  if df[col].dtypes == 'object':         
    encoder = LabelEncoder()         
    df[col] = encoder.fit_transform(df[col])



