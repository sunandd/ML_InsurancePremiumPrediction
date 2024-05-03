import sys
from logger import logging
from exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd

class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion=DataIngestion()
        self.data_transformation=DataTransformation()
        self.model_trainer=ModelTrainer()


    def run_pipeline(self):
        try:
            train_path, test_path=self.data_ingestion.initiate_data_ingestion()

            (
                train_arr,
                test_arr,
                preprocessor_file_path
            )= self.data_transformation.initiate_data_transformation(
                train_path=train_path, test_path=test_path
            )

        except Exception as e:
            raise CustomException(e, sys)    
                