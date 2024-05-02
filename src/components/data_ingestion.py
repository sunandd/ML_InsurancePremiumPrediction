import os
import sys
from logger import logging
from exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


#initialize the data ingetion  configuration

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts", "train.csv")
    test_data_path=os.path.join("artifacts", "test.csv")
    raw_data_path=os.path.join("artifacts", "raw_data.csv")

#create a data ingestion class
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the initiate_data_ingestion method of DataIngestion class")

            #read the data from the csv file
            df=pd.read_csv(os.path.join('notebook/data', 'insurance.csv'))
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info("Exited the initiate_data_ingestion method of DataIngestion class")

            logging.info("train test split")
            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path

            )


        except Exception as e:
            logging.info('error occured in data ingestion config')
            raise CustomException(e, sys)
        
      