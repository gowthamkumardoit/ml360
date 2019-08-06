from flask import Flask, jsonify, request
from flask_cors import CORS
import pyrebase
import pandas as pd
import simplejson as json

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
    "serviceAccount": "D:\\Kanini\\ml360-68ab3-firebase-adminsdk-07c2u-c215e020ee.json"
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
        a = json.loads((postData))
        print(a)
        url = (a['downloadURL'])
        data = pd.read_excel(url)
        prev = data.head()
        
       
    return jsonify({'cols': list(data.columns.values) , 'rows': prev.to_dict(orient='records')})


if __name__ == '__main__':
    app.run()
