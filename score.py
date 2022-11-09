import json
import numpy as np
import pandas as pd
import sklearn
import joblib
from azureml.core.model import Model

columns = ['passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 
           'precipTime', 'precipDepth', 'temperature']

def init():
    global model
    model_path = Model.get_model_path('nyc-taxi-fare')
    model = joblib.load(model_path)
    print('model loaded')

def run(input_json):
    # Get predictions and explanations for each data point
    inputs = json.loads(input_json)
    data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)
    # Make prediction
    predictions = model.predict(data_df)
    # You can return any data type as long as it is JSON-serializable
    return {'predictions': predictions.tolist()}