# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

def load_image(filename):
    img = Image.open(filename).resize((224, 224))
    img = img_to_array(img)
    img = img[:,:,0]
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

def run_example(filename):
    img = load_image(filename)
    model = tf.keras.models.load_model('/content/drive/MyDrive/Models/saved_fashion.h5')
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

def main():
    run_example('/content/kiki_fashion_example.jpg')

if __name__ == "__main__":
    main()
