"""The main processing engine for HA 3D Blueprint."""
from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Blueprint Engine is running."}

@app.post("/tag_location")
def tag_location(data: dict):
    print(f"Received data: {data}")
    # TODO: Store data in InfluxDB
    # TODO: Run tomography algorithm
    return {"status": "received", "data": data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8124)