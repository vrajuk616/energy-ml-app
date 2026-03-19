import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model("energy_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# PREDICTION FUNCTION
# =========================
def predict_energy(temp, humidity, wind, solar):
    try:
        data = np.array([[temp, humidity, wind, solar]])
        data = scaler.transform(data)
        prediction = model.predict(data)[0][0]
        return f"⚡ Predicted Energy Consumption: {round(prediction, 2)}"
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# UI DESIGN
# =========================
iface = gr.Interface(
    fn=predict_energy,
    inputs=[
        gr.Number(label="🌡 Temperature"),
        gr.Number(label="💧 Humidity"),
        gr.Number(label="🌬 Wind Speed"),
        gr.Number(label="☀ Solar Irradiance")
    ],
    outputs="text",
    title="⚡ Energy Consumption Predictor",
    description="Enter environmental values to predict energy usage"
)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
