import streamlit as st
import cv2
import json
from keras.models import load_model
#from tensorflow import keras as  tf_keras
from tensorflow.python.layers.normalization import BatchNormalization
import numpy as np

# Load data
with open("fyp.json") as f:
    data = json.load(f)
Disease_data = list(data)

# Load model
model_result = load_model("Pnemonia_model.h5", compile=True)

def Penumonia_prediction(image):
    image = cv2.resize(image, (224, 224))
    result = model_result.predict(image.reshape(1, 224, 224, 3))

    disease_name = Disease_data[result.argmax()]
    description = data[disease_name]['description']
    symptoms = data[disease_name]['symptoms']
    causes = data[disease_name]['causses']
    treatment = data[disease_name]['treatment']

    return disease_name, description, symptoms, causes, treatment

st.title("Pneumonia Prediction")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        with st.spinner('Predicting...'):
            disease_name, description, symptoms, causes, treatment = Penumonia_prediction(image)
        st.success('Prediction Done!')

        st.subheader("Prediction Results:")
        st.write("Disease Name:", disease_name)
        st.write("Description:", description)
        st.write("Symptoms:", symptoms)
        st.write("Causes:", causes)
        st.write("Treatment:", treatment)
