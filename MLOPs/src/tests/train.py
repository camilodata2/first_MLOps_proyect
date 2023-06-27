from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from MLOPs.src.tests.utils import *
from sklearn.preprocessing import StandardScaler
import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading Data...')
data = pd.read_csv('data/adult.csv')

logger.info('Loading model...')
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean',missing_values=np.nan)),
    ('radom_forest', RandomForestClassifier())
])

logger.info('Seraparating dataset into my target variable and indepent variable')
X=data.iloc[:,0:14].values
y=data.iloc[:,-1].values

logger.info('we take a sub-sample of the dataset')
selector = ExtraTreesClassifier(random_state = 42)
selector.fit(X, y)
feature_imp = selector.feature_importances_
for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2)) ,

logger.info('Here we gonna StandarScaler our dataser')
X=data.drop(['workclass', 'education', 'race', 'sex', 'capital.loss', 'native.country'],axis=1)
for col in X.columns:
    scaler=StandardScaler()
    X[col]=scaler.fit_transform(X[col].values.reshape(-1.1))

logger.info('balancing the data set')
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = 42)
ros.fit(X, y)
X_resampled, y_resampled = ros.fit_resample(X, y)
round(y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'

logger.info('split data into traning and test')
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42)


logger.info('Setting Hyperparameter to tune')
n_estimators = [int(x) for x in np.linspace(start = 40, stop = 150, num = 15)]
max_depth = [int(x) for x in np.linspace(start=40, stop=150, num = 15)]
max_samples=[int(x) for x in np.linspace(start=4,stop=150,num=15)]
param_dist = {
    'n_estimators' : n_estimators,
    'max_depth' : max_depth,
    'max_samples':max_samples
}

logging.info('Algoritm of RF without tune extimator')
rf_tuned = RandomForestClassifier(random_state = 42)
rf_cv = RandomizedSearchCV(estimator = rf_tuned, param_distributions = param_dist, cv = 5, random_state = 42)


logger.info('Starting  RandomizedSearchCV search...')
rf_cv.fit(X_train, y_train)


logger.info('Cross validating with best model...')
final_result = cross_validate(rf_cv.best_score_, X_train, y_train, return_train_score=True, cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])
assert train_score > 0.81
assert test_score > 0.80

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

logger.info('Updating model...')
update_model(rf_cv.best_estimator_)

logger.info('Generating model report...')
validation_score = rf_cv.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, rf_cv.best_estimator_)

y_test_pred =rf_cv.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')