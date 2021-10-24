# Package imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump, load
import pandas as pd
import yaml
import logging
import argparse

class PrepModel:
    def __init__(self, input_path, output_path, label, params_path, pipe):
        self.data_input = pd.read_csv(input_path)
        self.output_path = output_path
        self.label = label
        self.params = fetch_params(params_path)
        self.pipe = pipe
        logger.info("Initialisation completed successfully!")

    def model_training(self):
        X = self.data_input
        y = X.pop(self.label)

        logger.info("Model training")
        model = self._model_pipeline().fit(X, y)

        logger.info("Model persistence")
        dump(model, output_path) 

    def _model_definition(self):
        return LogisticRegression(**params['model'])
    
    def _model_pipeline(self):
        return Pipeline([('preprocessor', self.pipe), ('model', self._model_definition())])