# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Define the function to load the fashion classification model
@st.cache(allow_output_mutation=True)
def load_fashion_model():
    model = tf.keras.models.load_model('/content/drive/MyDrive/Models/saved_fashion.h5')
    return model

# Define the function to preprocess and make predictions on uploaded images
def predict_fashion(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def main():
    model = load_fashion_model()  # Load the fashion classification model
    run_example('/content/kiki_fashion_example.jpg', model)  # Run the example using the loaded model

# Define the function to run the example
def run_example(filename, model):
    img = load_image(filename)
    if img is not None:
        result = np.argmax(model.predict(img), axis=1)
        if result == 0:
            st.write('Tshirt')
        elif result == 1:
            st.write('Top')
        elif result == 2:
            st.write('Pullover')
        elif result == 3:
            st.write('Dress')
        elif result == 4:
            st.write('Coat')
        elif result == 5:
            st.write('Sandal')
        elif result == 6:
            st.write('Shirt')
        elif result == 7:
            st.write('Snicker')
        elif result == 8:
            st.write('Bag')
        else:
            st.write('Ankle Boot')

# Define the function to load and preprocess the image
def load_image(filename):
    try:
        img = Image.open(filename).resize((224, 224))
        img = img_to_array(img)
        img = img[:,:,0]
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32')
        img = img / 255.0
        return img
    except FileNotFoundError:
        st.error("File not found. Please make sure the file path is correct.")
        return None

if __name__ == "__main__":
    main()
