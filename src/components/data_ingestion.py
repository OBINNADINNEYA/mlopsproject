import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logging

# DataIngestionConfig holds file paths for raw, train, and test data artifacts.
#using dataclass decorator allows us to bypass the constructor and assign paths to the variables as needed

from src.components.data_transformation import *
from src.components.model_trainer import *



@dataclass
class DataIngestionConfig:
    artifacts_dir: str = "artifacts"
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            # cross-platform path build
            src_csv = os.path.join("notebook", "data", "stud.csv")
            df = pd.read_csv(src_csv)
            logging.info("Read the dataset as dataframe")

            os.makedirs(self.ingestion_config.artifacts_dir, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train/test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is complete")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_data, test_data = DataIngestion().initiate_data_ingestion()

    train_arr,test_arr,preprocessor =  DataTrasformation().initiate_data_transformation(train_data,test_data)

    print(ModelTrainer().initiate_model_training(train_arr,test_arr,preprocessor))
