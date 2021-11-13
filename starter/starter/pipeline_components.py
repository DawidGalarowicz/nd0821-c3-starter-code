import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
from sklearn.metrics import fbeta_score, precision_score, recall_score
from helpers import fetch_params
import pandas as pd
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def data_pipeline():
    """Define ColumnTransformer that any raw data supplied to the model will be adjusted by"""
    logger.info("Preparing data pipeline")

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    numerical_features = [
        "age",
        "fnlgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    categorical_preproc = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
        ]
    )

    numerical_preproc = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_preproc", categorical_preproc, cat_features),
            ("numerical_preproc", numerical_preproc, numerical_features),
        ]
    )

    return preprocessor


def model_definition():
    """Specify the model that you want to train"""
    return LogisticRegression

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