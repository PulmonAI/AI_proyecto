from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
from tensorflow import keras


app = FastAPI()

class InputData(BaseModel):
    input_text: str

# Carga el modelo
modelo = tf.keras.models.load_model('modelo/modelo_entrenado.h5')

def preprocess(input_text):
    pass

def hacer_inferencia(input_text):
    global modelo

    input_text_preprocesado = preprocess(input_text)

    resultado = modelo.predict(input_text_preprocesado)

    resultado_procesado = postprocess(resultado)

    return resultado_procesado

@app.post("/predict/") # dentro de los parentesis va la ruta de la web
async def predict(input_data: InputData):
    input_text = input_data.input_text
    resultado_de_la_inferencia = hacer_inferencia(input_text)
    return {"result": resultado_de_la_inferencia}
