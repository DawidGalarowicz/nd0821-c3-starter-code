from joblib import dump
import pandas as pd
import logging
import argparse
from pipeline_components import data_pipeline
from helpers import fetch_params
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class PrepData:
    """Class to prepare data files used in modelling census data"""

    def __init__(self, input_path, output_path, params_path, pipe=data_pipeline()):
        self.params = fetch_params(params_path)
        self.target = self.params["data"]["target"]
        self.data_input = pd.read_csv(input_path, index_col=[0]).dropna(
            subset=[self.target]
        )
        self.output_path = output_path
        self.pipe = pipe
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self._data_splitting()
        logger.info("Initialisation completed successfully!")

    def preprocess(self, save=True):
        """Preprocess training data and save all data assets along with the fitted data pipeline"""
        logger.info("Preprocessing")

        train_output_clean = pd.concat([self.X_train, self.y_train], axis=1)
        test_output_clean = pd.concat([self.X_test, self.y_test], axis=1)

        fitted_pipeline = Pipeline([("preprocessor", self.pipe)]).fit(self.X_train)

        X_train_output_processed = pd.DataFrame(
            fitted_pipeline.transform(self.X_train), index=self.X_train.index
        )
        train_output_processed = pd.concat(
            [X_train_output_processed, self.y_train], axis=1
        )

        logger.info("Saving data outputs")

        if save:
            train_output_processed.to_csv(
                os.path.join(self.output_path, "train_processed_census.csv")
            )
            train_output_clean.to_csv(
                os.path.join(self.output_path, "train_clean_census.csv")
            )
            test_output_clean.to_csv(
                os.path.join(self.output_path, "test_clean_census.csv")
            )
            dump(
                fitted_pipeline,
                os.path.join(self.output_path, "fitted_data_pipeline.joblib"),
            )

            return None

        else:
            return train_output_processed, train_output_clean, test_output_clean

    def _data_splitting(self):
        """Create train and test sets"""
        logger.info("Creating train/test sets")

        label_mapping = {"<=50K": 0, ">50K": 1}
        self.data_input.loc[:, self.target] = self.data_input[self.target].map(
            label_mapping
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.data_input.drop(columns=[self.target]),
            self.data_input[[self.target]],
            random_state=self.params["seed"],
            test_size=self.params["data"]["test_size"],
        )

        self.X_train = self.X_train.append(X_train)
        self.X_test = self.X_test.append(X_test)

        self.y_train = self.y_train.append(y_train)
        self.y_test = self.y_test.append(y_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creating data assets for the project")

    parser.add_argument("--input_path", type=str, help="Path to the raw dataset")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where the assets should be stored",
    )

    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to your params.yaml created for DVC",
        default="params.yaml",
        required=False,
    )

    args = parser.parse_args()

    PrepData(args.input_path, args.output_path, args.params_path).preprocess()
