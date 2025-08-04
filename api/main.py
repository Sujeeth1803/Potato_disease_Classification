from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL=tf.keras.models.load_model("../saved_models/1.keras")
Classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")

async def ping():
    return {"message": " ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    image = np.array(Image.open(BytesIO(file_bytes)))
    predicted = MODEL.predict(np.expand_dims(image, 0))
    predicted_class = Classes[np.argmax(predicted[0])]
    confidence_score = np.max(predicted[0])
    return {"class": predicted_class, "confidence_score": float(confidence_score)}



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)