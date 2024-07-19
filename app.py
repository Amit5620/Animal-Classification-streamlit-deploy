import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

IMAGE_SIZE = (180, 180)
THRESHOLD = 0.5

st.set_page_config(
    page_title="Animal Classification",
    page_icon="👨‍⚕️"
)

st.header(':blue[Welcome to our Animal Classification website.] :syringe: :pill: :ambulance:', divider='rainbow')

placeholder = st.empty()

with placeholder.form("upload"):
    st.header(':yellow[Provide an Animal image.]:mag:', divider='orange')
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit = st.form_submit_button("Predict")

if submit and uploaded_file:
    placeholder.empty()

    # img_array = tf.keras.preprocessing.image.img_to_array(image)
    # img_array = tf.expand_dims(img_array, 0)

    img = Image.open(uploaded_file)
    image = np.array(img)
    resized_image = cv2.resize(image, (180,180))
    resized_image = resized_image.reshape(-1,180,180,3)
    resized_image = np.array(resized_image)

    model = tf.keras.models.load_model('./models/animals.keras')
    predictions = model.predict(resized_image)
    max_index = predictions.argmax()

    labels = ["Bird", "Cat", "Dog"]
    label = labels[max_index]
    prediction = predictions[0][max_index]

    if prediction < THRESHOLD:
        label = "Unknown"
        prediction = 0.0

    prediction = round(prediction * 100, 2)

    st.header(f':mag: :rainbow[This is a : {label}]')
    st.write(f'Confidence {prediction}%.')

    st.image(img, caption='Uploaded Image')


