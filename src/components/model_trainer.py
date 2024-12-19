import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utills import save_obj
from src.utills import evaluate_model

import os, sys
from dataclasses import dataclass

# Model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE



## Model Trainning Configuration

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

## Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config  = ModelTrainerconfig()

    def  initiate_model_training(self,train_arr, test_arr):
        try:
            logging.info("Splitting independent and Dependent Variable")
            X_train,y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)

            ## Train multiple models

            models = {

                "LogisticRegression":LogisticRegression(),
                "DecisionTreeClassifier":DecisionTreeClassifier(),
                "RandomForestClassifier":RandomForestClassifier(),
                "Extratressclassifier":ExtraTreesClassifier(),
                "KNeighborsClassifier":KNeighborsClassifier(n_neighbors=5),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "GradientBoostingClassifier":GradientBoostingClassifier(),
                "SVC":SVC()
                }

            model_report : dict = evaluate_model(X_resampled_smote,y_resampled_smote, X_test, y_test, models)
            print(model_report)
            print("\n ==================================================================================")
            logging.info(f"Model report info : {model_report}")

            ## To get best model from model dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best model found , Best model name is {best_model_name} and that Accuracy Score: {best_model_score}")
            print("\n=================================================================")
            logging.info(f"Best model found , Best model name is {best_model_name} and that Accuracy Score: {best_model_score}")


            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except   Exception as e:
            logging.info("Error occured in model trainer path") 
            raise CustomException(e, sys)
        
