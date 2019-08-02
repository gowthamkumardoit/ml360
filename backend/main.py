import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate(
    "D:\\Kanini\\ml360-68ab3-firebase-adminsdk-07c2u-6036fcc276.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'gs://ml360-68ab3.appspot.com' })
bucket = storage.bucket()
blob = bucket.blob('file_279939')
blob.upload_from_filename('uploads/file_279939')
print(blob.public_url)

# import pyrebase
# import pandas as pd
# config = {
#     "apiKey": "AIzaSyCey6MlLMQlBR15Nl7cSkFlEDT-lNkKWlA",
#     "authDomain": "ml360-68ab3.firebaseapp.com",
#     "databaseURL": "https://ml360-68ab3.firebaseio.com",
#     "projectId": "ml360-68ab3",
#     "storageBucket": "ml360-68ab3.appspot.com",
#     "messagingSenderId": "86402740248",
#     "appId": "1:86402740248:web:8caa6dbf2183ec79",
#     "serviceAccount": "D:\\Kanini\\ml360-68ab3-firebase-adminsdk-07c2u-6036fcc276.json"
# }

# firebase = pyrebase.initialize_app(config)

# db = firebase.storage()
# path = db.child('uploads/file_279939').path
# print(path)
# # url = db.child('uploads/file_279939').get_url('9b956592-8494-459c-920b-4e2cef9047ba')
# # print(url)

# # import csv
# # import urllib3
# # import chardet
# # http = urllib3.PoolManager()

# # # r = http.request('GET', url)
# # import pandas as pd
# # import io
# # import requests

# # # url = "https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv"
# # s = requests.get(url).content
# # ds = pd.read_csv(io.StringIO(s.decode('utf-8')))
# # print(ds.describe())
