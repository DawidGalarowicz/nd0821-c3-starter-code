from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load
from pipeline_components import model_definition
from helpers import fetch_params
import pandas as pd
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class PrepModel:
    """Class to train the model and save it"""

    def __init__(self, input_path, output_path, params_path, pipe_path):
        self.data_input = pd.read_csv(input_path)
        self.output_path = output_path
        self.params = fetch_params(params_path)
        self.pipe = load(pipe_path)
        self.label = self.params["data"]["target"]
        logger.info("Initialisation completed successfully!")

    def model_training(self):
        X = self.data_input
        y = X.pop(self.label)

        logger.info("Model training")
        model = self._model_pipeline().fit(X, y)

        logger.info("Model persistence")
        dump(model, self.output_path + "/model.joblib")

    def _model_pipeline(self):
        return Pipeline(
            [
                ("preprocessor", self.pipe),
                ("model", model_definition()(**self.params["model"])),
            ]
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creating model assets for the project"
    )

    parser.add_argument("--input_path", type=str, help="Path to the cleaned dataset")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where the model should be stored",
    )

    parser.add_argument(
        "--pipe_path",
        type=str,
        help="Path to your fitted data pipeline",
        required=True,
    )

    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to your params.yaml created for DVC",
        default="params.yaml",
        required=False,
    )

    args = parser.parse_args()

    PrepModel(
        args.input_path, args.output_path, args.params_path, args.pipe_path
    ).model_training()
