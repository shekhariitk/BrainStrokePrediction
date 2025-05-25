import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utills import save_obj
from src.utills import evaluate_model

import os, sys
from dataclasses import dataclass

# Model 
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


from imblearn.over_sampling import SMOTE


## Model Training Configuration

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


## Model Training Class
class ModelTrainerClass:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting independent and Dependent Variable")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            # Resampling using SMOTE
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
            X_test_resampled_smote, y_test_resampled_smote = smote.fit_resample(X_test, y_test)

            ## Define models
            models = {
                   "LogisticRegression": LogisticRegression(),
                   "LogisticRegressionCV": LogisticRegressionCV(cv=5),
                   "DecisionTreeClassifier": DecisionTreeClassifier(),
                   "RandomForestClassifier": RandomForestClassifier(),
                   "ExtratreesClassifier": ExtraTreesClassifier(),
                   "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
                   "AdaBoostClassifier": AdaBoostClassifier(),
                   "GradientBoostingClassifier": GradientBoostingClassifier(),
                   "SVC": SVC(),
                   "XGBClassifier": XGBClassifier(),
                   "LGBMClassifier": LGBMClassifier(),
                   "CatBoostClassifier": CatBoostClassifier(silent=True)
}

            ## Define hyperparameter grids for each model
            param_grids = {
                "LogisticRegression": {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 200, 300]
                },
                "DecisionTreeClassifier": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "ExtratreesClassifier": {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                },
                "AdaBoostClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1]
                },
                "GradientBoostingClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 7]
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                "XGBClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "LGBMClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                "CatBoostClassifier": {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 7]
                },
                 "LogisticRegressionCV": {
                    'Cs': [0.1, 1, 10],
                     'cv': [3, 5]
            }
            }

            # Apply evaluate_model and get the model report
            model_report = evaluate_model(X_resampled_smote, y_resampled_smote, X_test_resampled_smote, y_test_resampled_smote, models, param_grids)
            print(model_report)
            print("\n ==================================================================================")
            logging.info(f"Model report info: {model_report}")

            # Use the function to select the best model based on recall, F1 score, and precision
            best_model = self.select_best_model_based_on_metrics(model_report, models)

            print(f"Best model found: {best_model}")
            logging.info(f"Best model found: {best_model}")

            # Save the best model
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error occurred in model trainer class")
            raise CustomException(e, sys)

    def select_best_model_based_on_metrics(self, model_report: dict, models: dict):
        # Select the best model based on recall first, then F1 score or precision as tie-breaker
        best_model_name = None
        best_recall = -1
        best_f1 = -1
        best_precision = -1
        best_model = None

        for model_name, metrics in model_report.items():
            recall = metrics['Recall']
            f1 = metrics['F1 Score']
            precision = metrics['Precision']

            # Prioritize recall first, then F1 score, then precision
            if recall > best_recall:
                best_recall = recall
                best_f1 = f1
                best_precision = precision
                best_model_name = model_name
                best_model = models[model_name]
            elif recall == best_recall:
                # If recall is tied, prioritize F1 score, then precision
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_model_name = model_name
                    best_model = models[model_name]
                elif f1 == best_f1:
                    # If F1 is tied, prioritize precision
                    if precision > best_precision:
                        best_precision = precision
                        best_model_name = model_name
                        best_model = models[model_name]

        print(f"Best model found: {best_model_name} with Recall: {best_recall}, F1 Score: {best_f1}, Precision: {best_precision}")
        logging.info(f"Best model found: {best_model_name} with Recall: {best_recall}, F1 Score: {best_f1}, Precision: {best_precision}")

        return best_model
