import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('shape_classifier.h5')
class_names = ['cube', 'pyramid', 'sphere']

# Streamlit app
st.title("3D Shape Classifier Web App")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None:
    # Ensure image is RGB (removes alpha channel if present)
    img = Image.open(uploaded_file).convert('RGB')

    # Show image
    st.image(img, caption='Uploaded image', use_container_width=True)

    # Prepare image for model
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Suggested orientation
    if predicted_class == 'pyramid':
        orientation = "Flat base down on Z axis"
    elif predicted_class == 'cube':
        orientation = "Any flat face down on bed"
    elif predicted_class == 'sphere':
        orientation = "Any orientation (use supports)"
    else:
        orientation = "Unknown"

    # Display results
    st.write(f"**Predicted:** {predicted_class} ({confidence:.1f}% confidence)")
    st.write(f"**Suggested print orientation:** {orientation}")
