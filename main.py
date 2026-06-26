from fastapi import FastAPI 
from pydantic import BaseModel
import joblib

app=FastAPI()
model=joblib.load('models/titanic_model.joblib')
scaler=joblib.load('models/titanic_scaler.joblib')
le=joblib.load('models/titanic_label.joblib')

SURVIVAL_LABELS = {
    0: "Not Survived",
    1: "Survived"
}

class Titanic(BaseModel):
    Age :float
    Pclass :int
    Fare :float
    Sex :str

@app.post('/prediction')
def predict(x:Titanic):
    x.Sex = x.Sex.strip().lower()
    x.Sex=le.transform([x.Sex])[0]
    sample=[[x.Age,x.Pclass,x.Fare,x.Sex]]
    sample=scaler.transform(sample)
    prediction=model.predict(sample)
    result=SURVIVAL_LABELS[prediction[0]]

    return{
        "prediction":result
    }