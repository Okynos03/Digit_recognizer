from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from predict import predict_digit

app = FastAPI()

class DigitInput(BaseModel):
    pixels: list  # 28x28 = 784 valores

@app.post("/predict")
def predict(input: DigitInput):
    arr = np.array(input.pixels, dtype=np.float32)
    pred = predict_digit(arr)
    return {"prediction": pred}
