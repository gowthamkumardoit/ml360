from flask import Flask, jsonify, request
from flask_cors import CORS
import pyrebase
import pandas as pd
import simplejson as json
import numpy as np
import Eda_imputation_new as eda
# import feature_selection as fs

app = Flask(__name__)
CORS(app)

config = {
    "apiKey": "AIzaSyCey6MlLMQlBR15Nl7cSkFlEDT-lNkKWlA",
    "authDomain": "ml360-68ab3.firebaseapp.com",
    "databaseURL": "https://ml360-68ab3.firebaseio.com",
    "projectId": "ml360-68ab3",
    "storageBucket": "ml360-68ab3.appspot.com",
    "messagingSenderId": "86402740248",
    "appId": "1:86402740248:web:8caa6dbf2183ec79",
    "serviceAccount": "E:\\ml360-68ab3-firebase-adminsdk-07c2u-a2fba5d0ff.json"
}

firebase = pyrebase.initialize_app(config)
db = firebase.storage()
files = db.list_files()


@app.route('/api',  methods=['POST'])
def hello():
    firebase = pyrebase.initialize_app(config)
    db = firebase.storage()

    if (request):
        postData = request.data
        res = json.loads((postData))
        url = (res['downloadURL'])

        if res['delimiter'] == 'comma':
            sep = ','
        elif res['delimiter'] == 'semicolon':
            sep = ';'
        elif res['delimiter'] == 'tab':
            sep = '\t'
        elif res['delimiter'] == 'pipe':
            sep = '|'

        if (res['extension'] == 'csv'):
            data = pd.read_csv(url, sep=sep)
            prev = data.fillna("NA")
            prev = prev.head()
        else:
            data = pd.read_excel(url)
            prev = data.fillna("NA")
            prev = prev.head()

        if (data.describe().to_dict(orient='records')):

            describe_rows = data.describe().to_dict(orient='records')
            describe_columns = list(data.describe().columns)
        else:
            describe_rows = ''
            describe_columns = ''

        df = data
        df = df.replace(' ', np.nan)
        df = df.replace('?', np.nan)
        df = df.replace('*', np.nan)
        df = df.replace('N.A.', np.nan)
        df = df.replace('NA.', np.nan)
        count_of_null = df.isnull().sum()
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame(
            {'percent_of_missing_values': percent_missing, 'count_of_missing_values': count_of_null})

        print('executed')
        return jsonify({
            'cols': list(data.columns.values),
            'rows': prev.to_dict(orient='records'),
            'summary_rows': describe_rows,
            'summary_cols': describe_columns,
            'na_data_rows': missing_value_df.to_dict(orient='records'),
            'yMax': len(df),
            'skew': df.skew(skipna=True).dropna().to_dict(),
            'kurtosis': df.kurtosis(skipna=True).dropna().to_dict()
        })


@app.route('/api/chart',  methods=['POST'])
def getChart():
    if (request):
        postData = request.data
        res = json.loads((postData))
        url = (res['downloadURL'])

        if res['delimiter'] == 'comma':
            sep = ','
        elif res['delimiter'] == 'semicolon':
            sep = ';'
        elif res['delimiter'] == 'tab':
            sep = '\t'
        elif res['delimiter'] == 'pipe':
            sep = '|'

        if (res['extension'] == 'csv'):
            data = pd.read_csv(url, sep=sep)
            print(data)
        else:
            data = pd.read_excel(url)

        items = data[res['chart_column']].astype(float)
        items = items.fillna(0)
        outliers = detect_outliers(items)

        print(outliers)
        return jsonify({
            'columns': list((items.values)),
            'outliers': list(outliers)
        })

def detect_outliers(x):
   outliers = []
   q1 , q3 = np.percentile(x,[25,75])
   iqr = q3 - q1
   lower_bound = q1 - (1.5 * iqr)
   upper_bound = q3 + (1.5 * iqr)
   for i in x:
       if i < lower_bound or i > upper_bound:
           outliers.append(i)
   return outliers 

@app.route('/api/imputedValues',  methods=['POST'])
def getImputedResult():
     if (request):
        postData = request.data
        res = json.loads((postData))
        url = (res['downloadURL'])
        targetType= (res['targetType'])

        if res['delimiter'] == 'comma':
            sep = ','
        elif res['delimiter'] == 'semicolon':
            sep = ';'
        elif res['delimiter'] == 'tab':
            sep = '\t'
        elif res['delimiter'] == 'pipe':
            sep = '|'

        if (res['extension'] == 'csv'):
            data = pd.read_csv(url, sep=sep)
            print(data)
        else:
            data = pd.read_excel(url)

        treatedTypesList, null_treated_df = eda.imputation(data)
        print(null_treated_df.info())
        feature_columns, feature_columns_values = eda.feature_selection(null_treated_df, res['targetColumn'], targetType)
        print('Final Data for Modelling')
        return jsonify({
            'treatedTypesList': treatedTypesList,
            'feature_columns': feature_columns,
            'feature_columns_values': feature_columns_values
        })

@app.route('/api/models/regressor/linear',  methods=['POST'])
def linear():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
    linear_regression_outcomes = eda.LinearRegression_modelling(target)
    return linear_regression_outcomes

@app.route('/api/models/regressor/random-forest',  methods=['POST'])
def RandomForestRegressor():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
    random_forest_regression_outcomes = eda.RandomForest_modelling_regressor(target)
    return random_forest_regression_outcomes

@app.route('/api/models/regressor/knn',  methods=['POST'])
def KNN():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
    knn_regression_outcomes = eda.KNN_modelling_scaled(target)
    return knn_regression_outcomes


@app.route('/api/models/classifier/logistic',  methods=['POST'])
def Logistic():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
        logistic_outcomes = eda.LogisticRegression_modelling(target)
        return logistic_outcomes

@app.route('/api/models/classifier/random-forest',  methods=['POST'])
def RandomForestModelling():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
        random_forest_modelling_outcomes = eda.RandomForest_modelling(target)
        return random_forest_modelling_outcomes

@app.route('/api/models/classifier/gb',  methods=['POST'])
def GB():
    if (request):
        postData = request.data
        res = json.loads((postData))
        target = (res['target'])
        gb_outcomes = eda.GB_modelling(target)
        return gb_outcomes

if __name__ == '__main__':
    app.run()
