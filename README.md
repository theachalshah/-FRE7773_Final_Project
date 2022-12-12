# -FRE7773_Final_Project
Machine Learning in Finance(MLOps)  - Weekly Buy Sell(Long Short) signals for S&P 500

Deployment Link (Google Cloud Platform) - https://mlfin-370922.ue.r.appspot.com

The [Google Cloud Platform][1] contains all files for deployment. 

[1]:https://github.com/theachalshah/-FRE7773_Final_Project/tree/main/Google%20Cloud%20Platform "Google Cloud Platform"

Executing Project
Make sure all packages in requirements.txt are installed to the correct version
pip install -r requirements.txt
Put the engineered and raw data under the ./data folder

Run my_flow.py under the right directory to generate the metadata folder

python3 my_flow.py run
After running my_flow.py, run app.py to generate a web application that allows users to input stock id and time id to make a prediction on volatility
python3 app.py run
Run model_selection.ipynb to get the overview of the pipeline of the project
