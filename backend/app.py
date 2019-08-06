from flask import Flask, jsonify, request
from flask_cors import CORS
import pyrebase
import pandas as pd
import simplejson as json
import numpy as np

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
        if (res['extension'] == 'csv'):
            data = pd.read_csv(url, sep=";")
            prev = data.head()
        else:
            data = pd.read_excel(url)
            prev = data.head()

        if (data.describe().to_dict(orient='records')):
            describe_rows = data.describe().to_dict(orient='records')
            describe_columns = list(data.describe().columns)
        else:
            describe_rows = ''
            describe_columns = ''

        df = data
        df = df.replace('?', np.nan)
        df = df.replace('*', np.nan)
        df = df.replace('N.A.', np.nan)
        df = df.replace('NA.', np.nan)
        count_of_null = df.isnull().sum()
        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame(
            {'percent_of_missing_values': percent_missing, 'count_of_missing_values': count_of_null})

        print(df.skew(skipna=True).dropna())
        print(df.kurtosis(skipna=True).dropna())
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


if __name__ == '__main__':
    app.run()
