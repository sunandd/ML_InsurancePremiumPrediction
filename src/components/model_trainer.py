import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.utils import evaluate_models
from src.utils import save_object
from src.utils import load_object
from src.utils import upload_file

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "Random Forest Regression": RandomForestRegressor(),
                "Support Vector Regression": SVR()
                }
            
            logging.info(f"Extracting model config file path")

            model_report: dict = evaluate_models(x=x_train, y=y_train, models=models)

            ## To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from  dictionary

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # To get best model score extract
            if best_model_score < 0.6:
                raise Exception("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')



            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )


        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)    

 