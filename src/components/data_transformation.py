from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import sys, os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utills import save_obj

# Data Transformation Configuration
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")

# Data Transformation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            numerical_columns = ['age', 'avg_glucose_level', 'bmi']
            categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(sparse=False, drop="first")),
                    ('scalar', StandardScaler(with_mean=False))
                ]
            )

            # Column transformer that applies the respective pipelines
            processor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            logging.info("Pipeline completed")
            return processor

        except Exception as e:
            logging.error(f"Error in creating data transformation pipeline: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading of train and test data has been completed")
            logging.info(f"Train DataFrame Head: \n {train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n {test_df.head().to_string()}")

            # Obtain preprocessor object (ColumnTransformer)
            preprocessor_Obj = self.get_data_transformation_object()

            # Target column
            target_column = 'stroke'

            # Exclude the target column (stroke) from features
            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            # Applying transformations
            input_feature_train_arr = preprocessor_Obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_Obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessor object to the train and test datasets")

            # Concatenating the transformed features with target values
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Saving the preprocessor object
            save_obj(file_path=self.data_transformation_config.preprocessor_ob_file_path, obj=preprocessor_Obj)

            logging.info("Preprocessor is created and saved")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {str(e)}")
            raise CustomException(e, sys)

        

     
