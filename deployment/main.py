from fastapi import Body, FastAPI
import uvicorn
import torch
from pydantic import BaseModel
from typing import Annotated

from utils import Utils

pred_examples = Utils.get_examples()


class Readings(BaseModel):
    sensor1_sensor2: list[list[float]]  # list of pairs of values sensor1-sensor2


class Pred(BaseModel):
    speed: float  # predicted speed


model = Utils.get_model()
app = FastAPI()


@app.get("/")
async def root():
    return {
        "Name": "Gearbox Speed Estimation",
        "Description": "Predicts shaft speed. Requires 12500 readings pairs([sensor1, sensor2]). Missed readings are treated as 0s",
    }


@app.post("/predict/", response_model=Pred)
def predict(readings: Annotated[Readings, Body(openapi_examples=pred_examples)]):
    model.eval()
    with torch.no_grad():
        input = Utils.prepare_data(readings)
        pred = model(input).flatten().to("cpu")  # prediction

        return Pred(speed=pred)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
