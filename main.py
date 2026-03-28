from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

# модель жана scaler жүктөө
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class PersonSchema(BaseModel):
    p_class: int
    gender: str
    age: int
    sib_sp: int
    parch: int
    fare: float
    embarked: str
    # Эгер башка categorical болсо, аны дагы кошсо болот

@app.post("/predict")
async def predict(person: PersonSchema):
    person_dict = person.dict()

    # gender encode
    gender_female = 1 if person_dict['gender']=='female' else 0

    # embarked encode (dummy)
    embarked_C = 1 if person_dict['embarked']=='C' else 0
    embarked_Q = 1 if person_dict['embarked']=='Q' else 0
    embarked_S = 1 if person_dict['embarked']=='S' else 0

    # Бул жерде тренингдеги башка categorical features encode кылынуусу керек
    # Эгер trainда 22 feature болсо, missing features үчүн 0 кошобуз
    # Мисалы:
    extra_features = [0]*(22 - 5 - 1 - 3)
    # 5: p_class, age, sib_sp, parch, fare
    # 1: gender_female
    # 3: embarked_C, Q, S
    # Калганы missing (trainда encode кылган categorical)

    # Person data order тренингдеги order менен дал келиши керек
    person_data = [
        person_dict['p_class'],
        person_dict['age'],
        person_dict['sib_sp'],
        person_dict['parch'],
        person_dict['fare'],
        gender_female,
        embarked_C,
        embarked_Q,
        embarked_S,
    ] + extra_features

    # transform жана predict
    scaler_data = scaler.transform([person_data])
    pred = model.predict(scaler_data)[0]

    return {"result": "alive" if int(pred)==1 else "not alive"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)