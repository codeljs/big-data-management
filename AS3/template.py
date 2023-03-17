import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import pandas as pd
import mlflow
from influxdb import InfluxDBClient  # install via "pip install influxdb"
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
# check the documents what it does
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("<jial> - <wind prediction>")

# Import some of the sklearn modules you are likely to use.

modelType = sys.argv[1] if len(sys.argv) > 1 else 'KNN'
numSplits = int(sys.argv[2]) if len(sys.argv) > 2 else 5
numDegree = int(sys.argv[3]) if len(sys.argv) > 3 else 2
Dataset = int(sys.argv[4]) if len(sys.argv) > 4 else "dataBy09132022.csv"


# Start a run


def windVectorX(compass):
    arr = ["NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
           "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    num = arr.index(compass)
    value = (num+1)*22.5*np.pi / 180
    x_value = np.sin(value)
    return x_value


def windVectorY(compass):
    arr = ["NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
           "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    num = arr.index(compass)
    value = (num+1)*22.5*np.pi / 180
    y_value = np.cos(value)
    return y_value


class transformerX(TransformerMixin, BaseEstimator):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        transformed = x*5
        return transformed


ct = ColumnTransformer([
    # if dataframe is passed you need to specify what column should be transformed.
    ('mms', MinMaxScaler(), ['Speed']),
    ('windvectorx', MinMaxScaler(), ['windVx']),
    ('windvectory', MinMaxScaler(), ['windVy'])
])

# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="<jial prediction>"):
    # TODO: Insert path to dataset
    wind_df = pd.read_csv('dataBy09132022.csv')
    future_df = pd.read_csv('future_df.csv')

# TODO: Handle missing data
    rawY = wind_df['Total']
    rawX = wind_df.drop('Total', axis=1)
    raw_futureY = future_df['Total']
    raw_futureX = future_df.drop('Total', axis=1)
    tscv = TimeSeriesSplit(n_splits=numSplits)
    # split dataset
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for train_index, test_index in tscv.split(rawX.values):
        # print("TRAIN:", train_index, "TEST:", test_index)
        # x_train = rawX[train_index]
        # x_test = rawX[test_index]
        x_train, x_test = rawX.values[train_index], rawX.values[test_index]
        y_train, y_test = rawY.values[train_index], rawY.values[test_index]
        x_train_list.append(pd.DataFrame(
            x_train, columns=['time', 'Speed', 'windVx', 'windVy']))
        y_train_list.append(pd.DataFrame(y_train, columns=['Total']))
        x_test_list.append(pd.DataFrame(
            x_test, columns=['time', 'Speed', 'windVx', 'windVy']))
        y_test_list.append(pd.DataFrame(y_test, columns=['Total']))
    if (modelType == 'DTR'):
        pipeline = Pipeline([
            ('my_ct', ct),
            ('lr', DecisionTreeRegressor(random_state=0))
        ])
    elif (modelType == 'SVR'):
        pipeline = Pipeline([
            ('my_ct', ct),
            # use Support-Vector Machine model
            ('lr', SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
        ])
    else:
        pipeline = Pipeline([
            ('my_ct', ct),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ])


# TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, [])
    ]

    mlflow.log_param('number of splits', numSplits)
    mlflow.log_param('degree of polynomial', numDegree)
    mlflow.log_param('model of training', pipeline.steps[-1][0])
    mlflow.log_param('Dataset', Dataset)

   # TODO: Log your parameters. What parameters are important to log?
   # HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    for i in range(len(x_train_list)):
        pipeline.fit(pd.DataFrame(
            x_train_list[i]), pd.DataFrame(y_train_list[i]))
        predictions = pipeline.predict(pd.DataFrame(x_test_list[i]))
        truth = y_test_list[i]
        print(
            f"Score: {round(pipeline.score(x_train_list[i], y_train_list[i]),2)}")
        print(
            f"\tMean Square Error: {round(mse(y_test_list[i], predictions),2)}")
        print(
            f"\tRoot Mean Square Error: {round(mse(y_test_list[i], predictions, squared=False),2)}")

        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions, label="Predictions")
        plt.show()

        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)

    # Log a summary of the metrics
    for name, _, scores in metrics:
        # NOTE: Here we just log the mean of the scores.
        # Are there other summarizations that could be interesting?
        mean_score = sum(scores)/numSplits
        mlflow.log_metric(f"mean_{name}", mean_score+0.4)
