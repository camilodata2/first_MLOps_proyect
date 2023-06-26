#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2018- 2023.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os


def joblib_save_model_to_dir(model, path):
    filename = path[path.rfind(os.path.sep)+1:] + '.pkl'
    os.makedirs(path)
    try:
        # note only up to scikit version 0.20.3
        from sklearn.externals import joblib
    except ImportError:
        # only for scikit 0.23.*
        import joblib
    joblib.dump(model, os.path.join(path, filename))


def create_scikit_learn_model_data(model_name='digits'):
    from sklearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn import preprocessing
    from sklearn import decomposition
    from sklearn import svm

    global model_data
    global model

    if model_name == 'digits':
        model_data = datasets.load_digits()
        scaler = preprocessing.StandardScaler()
        clf = svm.SVC(kernel='rbf')
        pipeline = Pipeline([('scaler', scaler), ('svc', clf)])
        model = pipeline.fit(model_data.data, model_data.target)
        predicted = model.predict(model_data.data[1: 10])
    if model_name == 'iris':
        model_data = datasets.load_iris()
        pca = decomposition.PCA()
        clf = svm.SVC(kernel='rbf')
        pipeline = Pipeline([('pca', pca), ('svc', clf)])
        model = pipeline.fit(model_data.data, model_data.target)
        predicted = model.predict(model_data.data[1: 10])

    return {
        'model': model,
        'pipeline': pipeline,
        'training_data': model_data.data,
        'training_target': model_data.target,
        'prediction': predicted
    }


def create_scikit_learn_model_directory(path):
    model_data = create_scikit_learn_model_data()
    joblib_save_model_to_dir(model_data['model'], path)


def create_scikit_learn_xgboost_model_data():
    from xgboost.sklearn import XGBClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_svmlight_file

    (X, y_train) = load_svmlight_file(os.path.join('svt', 'artifacts', 'agaricus.txt.train'))
    X_train = X.toarray()
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    # num_round = 2

    global model
    model = Pipeline([('scaler', StandardScaler()), ('classifier', XGBClassifier())])
    model.fit(X_train, y_train)
    predicted = model.predict(X_train[0:5, :])

    return {
        'model': model,
        'prediction': predicted
    }


def create_scikit_learn_xgboost_model_directory(path):
    model_data = create_scikit_learn_xgboost_model_data()
    joblib_save_model_to_dir(model_data['model'], path)


def create_spark_mllib_model_data():
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import StringIndexer, IndexToString, VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline

    spark = SparkSession.builder.getOrCreate()

    # df = spark.read.load(
    #     os.path.join(os.environ['SPARK_HOME'], 'data', 'mllib', 'sample_binary_classification_data.txt'),
    #     format='libsvm')

    df_data = spark.read \
        .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat') \
        .option('header', 'true') \
        .option('inferSchema', 'true') \
        .load("Cloud/artifacts/GoSales_Tx_NaiveBayes.csv")

    splitted_data = df_data.randomSplit([0.8, 0.18, 0.02], 24)
    train_data = splitted_data[0]
    test_data = splitted_data[1]
    predict_data = splitted_data[2]

    stringIndexer_label = StringIndexer(inputCol="PRODUCT_LINE", outputCol="label").fit(df_data)
    stringIndexer_prof = StringIndexer(inputCol="PROFESSION", outputCol="PROFESSION_IX")
    stringIndexer_gend = StringIndexer(inputCol="GENDER", outputCol="GENDER_IX")
    stringIndexer_mar = StringIndexer(inputCol="MARITAL_STATUS", outputCol="MARITAL_STATUS_IX")

    vectorAssembler_features = VectorAssembler(inputCols=["GENDER_IX", "AGE", "MARITAL_STATUS_IX", "PROFESSION_IX"],
                                               outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=stringIndexer_label.labels)
    pipeline_rf = Pipeline(stages=[stringIndexer_label, stringIndexer_prof, stringIndexer_gend, stringIndexer_mar,
                                   vectorAssembler_features, rf, labelConverter])
    model_rf = pipeline_rf.fit(train_data)

    return {
        'model': model_rf,
        'pipeline': pipeline_rf,
        'training_data': train_data,
        'test_data': test_data,
        'prediction': predict_data,
        'labels': stringIndexer_label.labels,
        #'output_schema': ''
    }


def create_spark_mllib_model_directory(path):
    model_data = create_spark_mllib_model_data()
    model_data['model'].write.overwrite.save(path)


def create_xgboost_model_data():
    import xgboost as xgb
    global agaricus
    agaricus = xgb.DMatrix(os.path.join('svt', 'artifacts', 'agaricus.txt.train'))
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 2

    global model
    model = xgb.train(params=param, dtrain=agaricus, num_boost_round=num_round)
    predicted = model.predict(agaricus.slice(range(5)))

    return {
        'model': model,
        'params': param,
        'prediction': predicted
    }


def create_xgboost_model_directory(path):
    model_data = create_xgboost_model_data()
    joblib_save_model_to_dir(model_data['model'], path)


def create_tensorflow_model_data():
    import tensorflow as tf
    import pandas as pd

    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    train_path = tf.keras.utils.get_file(
        "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file(
        "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    train_y = train.pop('Species')
    test_y = test.pop('Species')

    # The label column has now been removed from the features.
    train.head()

    import tensorflow as tf
    from sklearn.datasets import load_iris
    data = load_iris()

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(4,), name='my_input'),
        tf.keras.layers.Dense(28, activation=tf.nn.relu),
        tf.keras.layers.Dense(28, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    param = {
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy']
    }

    model.compile(**param)

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)

    predicted = model.predict(x_test[0:2])

    print(x_test)

    return {
        'model': model,
        'params': param,
        'prediction': predicted,
        'training_data': {'x_train': x_train, 'y_train': y_train},
        'test_data': {'x_test': x_test, 'y_test': y_test}
    }