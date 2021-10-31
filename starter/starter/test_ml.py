import pytest
import pandas as pd
from prepare_data import PrepData
import csv
import json
import os.path

@pytest.fixture(scope="session")
def input_data_path(tmpdir_factory):

    # Create a temporary directory and a file with a sample of random data
    testing_dir = tmpdir_factory.mktemp("data")
    fn = testing_dir.join("census_test_data.csv")

    header = [
        '','age','workclass','fnlgt',
        'education','education-num','marital-status',
        'occupation','relationship','race',
        'sex','capital-gain','capital-loss',
        'hours-per-week','native-country','salary']

    data = [
        [0,39,'State-gov',77516,'Masters',14,'Never-married','Exec-managerial','Own-child','White','Female',2174,41,50,'United-States','<=50K'],
        [1,32,'Private',197516,'Bachelors',13,'Never-married','Adm-clerical','Not-in-family','White','Male',3000,0,40,'United-States','<=50K'],
        [2,85,'Self-emp-not-inc',74516,'Bachelors',13,'Divorced','Adm-clerical','Not-in-family','White','Male',217,20,30,'Germany','>50K'],
        [3,23,'Federal-gov',72216,'HS-grad',9,'Never-married','Adm-clerical','Wife','Black','Female',27411,0,45,'United-States','<=50K'],
        [4,21,'Private',111111,'Bachelors',13,'Married-civ-spouse','Prof-specialty','Not-in-family','White','Male',2144,34,40,'Honduras','>50K'],
        [5,44,'Private',214123,'Masters',14,'Never-married','Craft-repair','Husband','White','Male',20,0,40,'Mexico','<=50K']
    ]

    with open(fn, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header
        writer.writerow(header)

        # Write multiple rows
        writer.writerows(data)
    
    # Return the path
    return fn

@pytest.fixture(scope="session")
def input_params_path(tmpdir_factory):

    # Create a temporary directory and a file with parameters
    testing_dir = tmpdir_factory.mktemp("params")
    fn = testing_dir.join("example_params.json")

    params = {
        'seed': 909,
        'data': {'test_size': 0.2,
        'target': 'salary'}
    }

    # Store parameters in the file
    with open(fn, "w") as file:
        json.dump(params, file)
    
    # Return the path
    return fn

@pytest.fixture(scope="session")
def output_path(tmpdir_factory):

    # Create a temporary directory for outputs
    testing_dir = tmpdir_factory.mktemp("results")

    return testing_dir

def test_equal_rows_post_processing(input_data_path, output_path, input_params_path):
    train_output_processed, train_output_clean, test_output_clean = PrepData(input_data_path, output_path, input_params_path).preprocess(save = False)
    assert train_output_processed.shape[0] == train_output_clean.shape[0]

def test_correct_one_hot_encoding(input_data_path, output_path, input_params_path):
    train_output_processed, train_output_clean, test_output_clean = PrepData(input_data_path, output_path, input_params_path).preprocess(save = False)
    assert train_output_processed.shape[1] > train_output_clean.shape[1]

def test_saving_data(input_data_path, output_path, input_params_path):
    PrepData(input_data_path, output_path, input_params_path).preprocess()
    testing_dir = output_path
    assert os.path.exists(testing_dir +  '/train_processed_census.csv') and os.path.exists(testing_dir + '/train_clean_census.csv') and os.path.exists(testing_dir + '/test_clean_census.csv') 