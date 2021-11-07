from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import os
from joblib import load
import pandas as pd

path = Path(__file__).parent.absolute()
model = load(os.path.join(path, "model.joblib"))

example_header = [
    "",
    "age",
    "workclass",
    "fnlgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]

example_request = [
    0,
    39,
    "State-gov",
    77516,
    "Masters",
    14,
    "Never-married",
    "Exec-managerial",
    "Own-child",
    "White",
    "Female",
    2174,
    41,
    50,
    "United-States",
]

example = dict(zip(example_header, example_request))

example_full = {"example": example}

app = FastAPI()


class Census(BaseModel):
    age: float
    workclass: str
    fnlgt: float
    education: str
    education_num: float
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: str

    class Config:
        schema_extra = example_full


@app.get("/")
async def root():
    return "Hello World! Welcome to my prediction app!"


@app.post("/predict")
async def pred_function(census_data: Census):
    mydict = dict(
        (k.replace("_", "-") if "_" in k else k, [v])
        for k, v in census_data.dict().items()
    )
    data = pd.DataFrame.from_dict(mydict)
    pred = model.predict(data)
    response_positive = "Person will earn at least $50,000"
    response_negative = "Person will earn below $50,000"
    return response_positive if pred == 1 else response_negative
