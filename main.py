from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel


model = joblib.load("model (7).pkl")
scaler = joblib.load("scaler (8).pkl")


titanic_app = FastAPI()

class PersonSchema(BaseModel):
    p_class: int
    gender: str
    age: int
    sib_sp: int
    parch: int
    fare: float
    embarked: str


@titanic_app.post("/predict")
async def predict(person: PersonSchema):
    person_dict = person.dict()

    gender = person_dict.pop("gender")
    gender1_0 = [
        1 if gender == "female" else 0,
    ]

    embarked = person_dict.pop('embarked')
    embarked1_0 = [
        1 if embarked == 'Q' else 0,
        1 if embarked == 'S' else 0,
    ]

    person_df = list(person_dict.values()) + gender1_0 + embarked1_0
    scaler_df = scaler.transform([person_df])
    pred = model.predict(scaler_df)[0]
    return {"result": "alive" if  pred == 1 else "not alive"}


if __name__ == "__main__":
    uvicorn.run(titanic_app, host="127.0.0.1", port=8001)