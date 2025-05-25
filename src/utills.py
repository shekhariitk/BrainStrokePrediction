import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e: 
        raise CustomException(e, sys)   
    


def evaluate_model(X_train, y_train, X_test, y_test, models, param_grids):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param_grid = param_grids[list(models.keys())[i]]  # Get the corresponding parameter grid

            logging.info(f"model:{model} is started")

            # Set up GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
            
            # Fit the model using GridSearchCV
            grid_search.fit(X_train, y_train)


            model.set_params(**grid_search.best_params_)
            model.fit(X_train,y_train)

            logging.info(f"model:{model} is Evaluated and best param is {grid_search.best_params_}")

            # Make Predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy, precision, recall, and F1 score
            test_model_score = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # Add the metrics to the report for this model
            report[list(models.keys())[i]] = {
                'Accuracy': test_model_score,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Classification Report': classification_report(y_test, y_pred)
            }
        
        return report
    
    except Exception as e:
        print(f"Error: {e}")

    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e: 
        logging.info("Error Occured during load object ")
        raise CustomException(e, sys)
