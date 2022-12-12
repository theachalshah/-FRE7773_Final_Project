from flask import Flask, render_template, request
import numpy as np
from metaflow import Flow
from metaflow import get_metadata, metadata

#### THIS IS GLOBAL, SO OBJECTS LIKE THE MODEL CAN BE RE-USED ACROSS REQUESTS ####

FLOW_NAME = 'LongShort' # name of the target class that generated the model
# Set the metadata provider as the src folder in the project,
# which should contains /.metaflow
metadata('../ML_PROJ')
# Fetch currently configured metadata provider to check it's local!
print(get_metadata())

def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

# get artifacts from latest run, using Metaflow Client API
latest_run = get_latest_successful_run(FLOW_NAME)
latest_model1 = latest_run.data.model1
latest_model2 = latest_run.data.model2

# We need to initialise the Flask object to run the flask app 
# By assigning parameters as static folder name,templates folder name
app = Flask(__name__, static_folder='static',template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_signal():
    Open = float(request.form.get('Open'))
    High = float(request.form.get('High'))
    Low = float(request.form.get('Low'))
    Adj_Close = float(request.form.get('Adj Close'))
    Volume = float(request.form.get('Volume'))
    atr = float(request.form.get('atr'))
    rsi = float(request.form.get('rsi'))
    obv = float(request.form.get('obv'))
    wcp = float(request.form.get('wcp'))
    trend = float(request.form.get('trend'))
    kalf = float(request.form.get('kalf'))
    
    # prediction
    result1 = latest_model1.predict(np.array([Open,High,Low,Adj_Close,Volume,atr,rsi,obv,wcp,trend,kalf]).reshape(1,11))
    result2 = latest_model2.predict(np.array([Open,High,Low,Adj_Close,Volume,atr,rsi,obv,wcp,trend,kalf]).reshape(1,11))
    signal = result1+result2
    if signal == 2:
        signal = 'Buy'
  
    if signal == -2:
        signal = 'Sell'
    
    if signal == 0:
        signal = 'No Opportunity'

    return render_template('index.html',signal=signal)
    

if __name__=='__main__':
  # Run the Flask app to run the server
  app.run(debug=True)