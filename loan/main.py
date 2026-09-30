from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("model_trained.pk3")
app.mount("/static", StaticFiles(directory="static"), name="static")
class LoanInput(BaseModel):
    Age: int
    Salary: float
    Credit_Score: int
    Loan_Amount: float
    Loan_Term: str
    Employment_Status: str
    Residence_Type: str
    Previous_Default: str


@app.get("/")
def home():
    return {"Message": "Loan Approval Prediction API"}


@app.post("/predict")
def prediction(data: LoanInput):

    input_df = pd.DataFrame({
        "Age": [data.Age],
        "Salary": [data.Salary],
        "Credit_Score": [data.Credit_Score],
        "Loan_Amount": [data.Loan_Amount],
        "Loan_Term": [data.Loan_Term],
        "Employment_Status": [data.Employment_Status],
        "Residence_Type": [data.Residence_Type],
        "Previous_Default": [data.Previous_Default]
    })

    prediction = model.predict(input_df)

    result = "Loan Approved" if prediction[0] == "Yes" else "Loan Rejected"

    return {
        "Prediction": result
    }