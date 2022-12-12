from flask import Flask,render_template,request
import pickle
import numpy as np

model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))

app = Flask(__name__)

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
    result1 = model1.predict(np.array([Open,High,Low,Adj_Close,Volume,atr,rsi,obv,wcp,trend,kalf]).reshape(1,11))
    result2 = model2.predict(np.array([Open,High,Low,Adj_Close,Volume,atr,rsi,obv,wcp,trend,kalf]).reshape(1,11))
    
    signal = result1+result2
    if signal == 2:
        signal = 'Buy'
  
    if signal == -2:
        signal = 'Sell'
    
    if signal == 0:
        signal = 'No Opportunity'

    

    return render_template('index.html',signal=signal)


if __name__ == '__main__':
    app.run(debug=True)