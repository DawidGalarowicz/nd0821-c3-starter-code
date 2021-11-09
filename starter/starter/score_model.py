from joblib import load
from sklearn.metrics import fbeta_score, precision_score, recall_score
from helpers import fetch_params
import pandas as pd
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class ScoreModel:
    """Class to quantify model performance"""

    def __init__(self, input_path, output_path, params_path, model_path, variable):
        self.input_path = input_path
        self.variable = variable
        self.output_path = output_path
        self.params = fetch_params(params_path)
        self.model = load(model_path)
        self.label = self.params["data"]["target"]
        self.X = None
        self.y = None
        logger.info("Initialisation completed successfully!")

    def _generate_metrics(self):
        preds = self.model.predict(self.X)
        fbeta = fbeta_score(self.y, preds, beta=1, zero_division=1)
        precision = precision_score(self.y, preds, zero_division=1)
        recall = recall_score(self.y, preds, zero_division=1)
        return precision, recall, fbeta

    def performance_output(self):
        data = pd.read_csv(self.input_path)

        if self.variable is None:
            logger.info("Get results for the entire model")
            self.X = data.copy()
            self.y = self.X.pop(self.label)
            precision, recall, fbeta = self._generate_metrics()
            results = [{"precision": precision, "recall": recall, "fbeta": fbeta}]

        else:
            logger.info("Get results for your variable of choice")
            slices = data[self.variable].dropna().unique().tolist()
            results = []
            for slice in slices:
                self.X = data.copy()[data.copy()[self.variable] == slice]
                self.y = self.X.pop(self.label)
                precision, recall, fbeta = self._generate_metrics()
                results_iteration = [
                    {
                        "slice": slice,
                        "precision": precision,
                        "recall": recall,
                        "fbeta": fbeta,
                    }
                ]
                results.append(results_iteration)

        with open(self.output_path, "w") as f:
            print(results, file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to score your model")

    parser.add_argument("--input_path", type=str, help="Path to the cleaned dataset")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where the metrics should be stored",
    )

    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to your params.yaml created for DVC",
        default="params.yaml",
        required=False,
    )

    parser.add_argument(
        "--model_path", type=str, help="Path to your model", required=True,
    )

    parser.add_argument(
        "--variable",
        type=str,
        help="Name of the variable you want to generate performance slices for",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    ScoreModel(
        args.input_path,
        args.output_path,
        args.params_path,
        args.model_path,
        args.variable,
    ).performance_output()
