import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

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
        'age',
        'fnlgt',
        'education-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]

    categorical_preproc = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                                    ('encoder', OneHotEncoder())])

    numerical_preproc = Pipeline([('imputer', SimpleImputer(strategy="median")),
                                    ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
    transformers=[
        ("categorical_preproc", categorical_preproc, cat_features),
        ("numerical_preproc", numerical_preproc, numerical_features)
    ])

    return preprocessor

def model_definition():
    """Specify the model that you want to train"""
    return LogisticRegression