####################################################
## Getting the data

from influxdb import InfluxDBClient # install via "pip install influxdb"
import pandas as pd

client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

# Get the last 90 days of power generation data
generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    ) # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL


gen_df = get_df(generation)
wind_df = get_df(wind)


####################################################
## Preprocess the data / Compose your pipeline

from sklearn.pipeline import Pipeline

# Align the data frames

pipeline = Pipeline([
    # Here you can add your preproccesing transformers
    # And you can add your model as the final step
])

# Fit the pipeline

# Load stored model and compare with newly trained
# and store the best one

####################################################
## Do forecasting with the best one

# Get all future forecasts regardless of lead time
forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()"
    ) # Query written in InfluxQL
for_df = get_df(forecasts)

# Limit to only the newest source time
newest_source_time = for_df["Source_time"].max()
newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()

# Preprocess the forecasts and do predictions in one fell swoop 
# using your best pipeline.
pipeline.predict(newest_forecasts)