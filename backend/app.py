from flask import Flask, jsonify, request
from flask_cors import CORS
import pyrebase
import pandas as pd
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
    "serviceAccount": "E:\\ML360\\ml\\backend\\ml360-68ab3-firebase-adminsdk-07c2u-1d133345cf.json"
}

firebase = pyrebase.initialize_app(config)
db = firebase.storage()
files = db.list_files()
for file in files:
    print(db.child(file.name).get_url(None))

@app.route('/api',  methods=['POST'])
def hello():
    firebase = pyrebase.initialize_app(config)
    db = firebase.storage()
   
    if (request):
        postData = request.data
        url = (postData['downloadURL'])
        data = pd.read_excel(url)
        prev = data.head()
        print(prev)
        #url = db.child('uploads/'+postData['id']).get_url(postData.downloadURL)
        #print(url)
    return jsonify({'message': "Hello World!"})


if __name__ == '__main__':
    app.run()
