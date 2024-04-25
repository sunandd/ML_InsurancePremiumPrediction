
from pymongo.mongo_client import MongoClient
import pandas as pd
import json

uri = "mongodb+srv://sunandd:sunandd@cluster0.papwels.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

##create database Name
DATABASE_NAME="InsurancePremium"
COLLECTION_NAME="PremiumPrediction"
#read the data as a dataframe
df=pd.read_csv("datapatha")

#convert data into json
json_record=json.loads(df.to_json())

#dump the data to the database
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)