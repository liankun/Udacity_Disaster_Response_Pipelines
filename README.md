## Disaster Response Pipeline Project
There are two directory:
1. ETL_ML_Pipeline:
   - 'ETL Pipeline Preparation.ipynb': the notebook for ETL pipeline Preparation.In this notebook, the messages.csv data will be extracted, transformed , cleaned and finally loaded into a database
   - 'ML Pipeline Preparation.ipynb': load the data from the database (previous step), perform NLP clean, norm process and finally build machine learning model.

2. web_app: 
   put the notebooks from previous file into the web apps
   - web_app/app: 
     run.py will create the web application
     templates: necessary html templates to run web application
   - web_app/data:
     process_data.py python module to perform the ETL pipeline
   - web_app/models:
     train_classifier.py python module to perform ML pipeline


### Instructions:
1. Run the following commands in the project's web_app directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5001/

### Try on imbalanced dataset
The dataset used in this project is highly imbalanced, many categories have very bad f1 score. There are several ways to tackle this challenge:
1. Use different metric: for some category (like water,aid), recall is more important. we can increase the weight of the recal in metric
2. Try different algorithms: I use the adaBoost and Randomforest classifier in this project and adaBoost performs better
3. Undersampling: In this project I take the water category as a example and Undersample the dataset. The model works much better on this undersampled dataset, but when generalize to the whole dataset, the precision is very bad. This means the model underfit the generalized data.

In all, to solve the imbalanced dataset is not an easy task. I think one solution may be trying to use the dataset which has similar property. Besides, try to analysis the failed cases and extract some distinct pattern.  

### license
The code are free to use
