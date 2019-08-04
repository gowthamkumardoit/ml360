from flask import Flask, jsonify
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
    "serviceAccount": "D:\\Kanini\\ml360-68ab3-firebase-adminsdk-07c2u-6036fcc276.json"
}


@app.route('/api/<string:uid>/<string:filename>/',  methods=['POST', 'GET'])
def hello(uid, filename):
    firebase = pyrebase.initialize_app(config)
    db = firebase.storage()
    print(db.child('uploads/'+ uid + '/' + filename))
    return jsonify({'message': "Hello World!"})


if __name__ == '__main__':
    app.run()
