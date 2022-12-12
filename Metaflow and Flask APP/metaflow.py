# pylint: skip-file
from metaflow import FlowSpec, step, Parameter, IncludeFile, current, JSONType
from datetime import datetime
import os
from comet_ml import Experiment
from comet_ml.integration.metaflow import comet_flow
from io import StringIO
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import math 
from sklearn import metrics
import yfinance as yf
import pandas as pd
import numpy as np
import talib as ta
from pykalman import KalmanFilter
from fracdiff.sklearn import Fracdiff


assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


@comet_flow(project_name = "lS_Project")
class LongShort(FlowSpec):
    """
    LongShort implements a model to try and predict direction movement of stocks
    and recommend buy/sell/hold strategy.
    """
    @step
    def start(self) -> None:
        #Start up and set up the stock ticker, train and test dataset time period. 
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.train_start='1928-01-01'
        self.train_end='2008-12-31'
        self.test_start='2009-01-01'
        self.test_end='2021-12-31'
        self.tickers = ['^GSPC']
        self.next(self.load_data)

    @step
    def load_data(self) -> None: 
        #Load train and test dataset using yfinance
        self.train_df = yf.download(self.tickers,start=self.train_start,end=self.train_end,parse_dates=True,interval = "1wk")
        self.test_df = yf.download(self.tickers,start=self.test_start,end=self.test_end,parse_dates=True,interval = "1wk")
        self.datasets=[self.train_df,self.test_df]
        self.next(self.data_engineering,foreach='datasets')
    
    @step
    def data_engineering(self) -> None:
        #Feature engineering to extract technical indicator from the dataset
        df=self.input
        f = Fracdiff(0.3,window=7)
        a = f.fit_transform(df)
        statdf=pd.DataFrame(data = a, 
                        index = df.index, 
                        columns = df.columns)
        pred=statdf['Adj Close'].shift(-1)
        statdf['pred']=pred
        statdf.dropna(inplace=True)
        atr= ta.ATR(statdf['High'], statdf['Low'], statdf['Close'], timeperiod=7)
        rsi = ta.RSI(statdf['Close'], timeperiod=7)
        obv = ta.OBV(statdf['Close'], statdf['Volume'])
        wcp = ta.WCLPRICE(statdf['High'], statdf['Low'], statdf['Close'])
        trend = ta.HT_TRENDMODE(statdf['Close'])
        kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = statdf['Close'].values[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
        state_means,_ = kf.filter(statdf[['Close']].values)
        kalf = state_means.flatten()
        statdf['atr']=atr
        statdf['rsi']=rsi
        statdf['obv']=obv
        statdf['wcp']=wcp
        statdf['trend']=trend
        statdf['kalf']=kalf
        statdf['pred_change']=statdf['pred'].pct_change()
        statdf.dropna(inplace=True)
        mask_posbuy = (statdf.pred_change >= 0.00) 
        signalpredbuy= np.where(mask_posbuy, 1, 0)
        mask_possell = (statdf.pred_change < 0.00) 
        signalpredsell= np.where(mask_possell, -1, 0)
        statdf['signal']=signalpredbuy+signalpredsell
        self.statdf=statdf
        self.next(self.join)

    @step
    def join(self,inputs) -> None:
        #Join and separate train and test dataset
        self.results = [input.statdf for input in inputs]
        self.train, self.test = self.results[0],self.results[1]
        self.next(self.prepare_dataset)

    @step
    def prepare_dataset(self) -> None:
        #Prepare X_train, X_test, y_train and y_test datasets
        self.X_train = self.train[[ 'Open', 'High', 'Low', 'Adj Close', 'Volume','atr','rsi', 'obv', 'wcp', 'trend', 'kalf']]
        self.y_train = self.train['signal']
        self.X_test = self.test[[ 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'atr',
            'rsi', 'obv', 'wcp', 'trend', 'kalf']]
        self.y_test = self.test['signal']
        self.next(self.train_model)
        
    @step
    def train_model(self) -> None:
        #Train two separate models
        from sklearn.ensemble import AdaBoostClassifier
        from catboost import CatBoostClassifier

        self.model1=AdaBoostClassifier(random_state=6, n_estimators=45,learning_rate=0.025)
        self.model1.fit(self.X_train,self.y_train)
        self.model2 = CatBoostClassifier(iterations=100,learning_rate=0.025,random_state=6,verbose=2,max_depth=5)
        self.model2.fit(self.X_train,self.y_train)
        self.next(self.predictions)

    @step 
    def predictions(self) -> None:
        #Predict and log metrics
        from sklearn.metrics import precision_score, f1_score, recall_score

        experiment = Experiment(
            api_key="IWhOpkqIBwQzjeYqtvvU5caSS",
            project_name="lS_Project",
            workspace="nyu-fre-7773-2021",
            )
        self.y_pred1 = self.model1.predict(self.X_test)
        experiment.log_metric("Precision 1",precision_score(self.y_test,self.y_pred1, average = 'macro') )
        experiment.log_metric("F1 score 1",f1_score(self.y_test,self.y_pred1, average = 'macro') )
        experiment.log_metric("Recall score 1",recall_score(self.y_test,self.y_pred1, average = 'macro') )
        self.y_pred2 = self.model2.predict(self.X_test)
        experiment.log_metric("Precision 2", precision_score(self.y_test,self.y_pred2, average = 'macro'))
        experiment.log_metric("F1 score 2",f1_score(self.y_test,self.y_pred2, average = 'macro') )
        experiment.log_metric("Recall score 2",recall_score(self.y_test,self.y_pred2, average = 'macro') )
        experiment.end()
        self.next(self.signal_processing)

    @step 
    def signal_processing(self) -> None:
        #Aggregate information from both models to evaluate signal strength
        self.signal_strength = self.y_pred1+self.y_pred2
        self.signals=self.signal_strength.squeeze().tolist()
        self.next(self.signal_type)
     
    @step 
    def signal_type(self) -> None:
        #Convert predictions to 'Buy', 'Sell' and 'Do Nothing' signals for easy interpretation.
        signals=self.signals
        experiment = Experiment(
            api_key="IWhOpkqIBwQzjeYqtvvU5caSS",
            project_name="lS_Project",
            workspace="nyu-fre-7773-2021",
            )
        for i in range(len(signals)):
    
            if signals[i] == 2:
                signals[i] = 'buy'
  
            if signals[i] == -2:
                signals[i] = 'sell'
    
            if signals[i] == 0:
                signals[i] = 'do nothing'
        experiment.log_metric("signals",signals)
        self.next(self.end)
    
    @step
    def end(self) -> None:
        #End 
        print("Done at {}!".format(datetime.utcnow()))


if __name__ == '__main__':
    LongShort()
