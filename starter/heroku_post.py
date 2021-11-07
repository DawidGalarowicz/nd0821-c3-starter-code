import requests

data = {
    "": 0,
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Never-married",
    "occupation": "Exec-managerial",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Female",
    "capital_gain": 217400,
    "capital_loss": 41,
    "hours_per_week": 50,
    "native_country": "United-States"}

response = requests.post('https://dawid-udacity-project.herokuapp.com/predict', data = data)

print(response.status_code)
print(response.json())