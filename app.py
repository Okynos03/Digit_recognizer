from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from predict import predict_digit
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DigitInput(BaseModel):
    pixels: list

@app.post("/predict")
def predict(input: DigitInput):
    arr = np.array(input.pixels, dtype=np.float32)

    print("=== BACKEND PIXELS ===")
    print("max:", np.max(arr))
    print("min:", np.min(arr))
    print("sum:", np.sum(arr))
    print("=======================")

    pred = predict_digit(arr)
    return {"prediction": pred}