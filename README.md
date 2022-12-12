# -FRE7773_Final_Project
Machine Learning in Finance(MLOps)  - Weekly Buy Sell(Long Short) signals for S&P 500

Deployment Link (Google Cloud Platform) - https://mlfin-370922.ue.r.appspot.com

The [Google Cloud Platform][1] contains all files for deployment. 

[1]:https://github.com/theachalshah/-FRE7773_Final_Project/tree/main/Google%20Cloud%20Platform "Google Cloud Platform"

The [Presentation][4] explains the project theory and execution.
[4]:https://github.com/theachalshah/-FRE7773_Final_Project/blob/main/Machine%20Learning%20Project.pdf "Presentation"

The [file][3] contains data preprocessing, model training, testing.

[3]:https://github.com/theachalshah/-FRE7773_Final_Project/blob/main/Final_Weekly_MLProject.ipynb "file"
## Executing Project on local system

The [Metaflow and Flask APP][2] contains all files for deployment. 

[2]:https://github.com/theachalshah/-FRE7773_Final_Project/tree/main/Metaflow%20and%20Flask%20APP "Metaflow and Flask APPP"


Make sure all packages in requirements.txt are installed to the correct version

```
pip install -r requirements.txt
```

The data is taken directly from yahoo finance(yfinance api)

```
Run metaflow.py to execute metaflow 
```
```
python3 metaflow.py run
```

After running metaflow.py, run app.py to generate a web application that allows users to input price and technical indicators  to make a prediction for next week S&P 500 index. 

```
python3 app.py run
```

Click on predict to get BUY, SELL, NO OPPORTUNITY Singal
