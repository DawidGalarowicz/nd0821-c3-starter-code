# Package imports
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

class CensusModel:
    def __init__(self, input_path, output_path, params):
        self.input_path = pd.read_csv(args.input_path)
        self.output_path = args.output_path
        self.params = self._fetch_params()

    def data_cleaning(self):
        
        return params

    def _fetch_params(self):
        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)
        return params
    


# Use parameters tracked by DVC
with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

# Get the data
args.data_path


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
