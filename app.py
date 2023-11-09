import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import model2

class_names = model2.class_names

model = tf.keras.models.load_model('my_model.hdf5')
st.write("""
         # Sign Lasnguage Alphabet Letter Prediction
         """
         )
st.write("This is a simple ASL hand sign classification web app to predict the alpgabetical value")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    
        size = (200,200)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        
    
        prediction = model.predict(img_array)
        
        return prediction
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    
    score = tf.nn.softmax(predictions[0])
    print(class_names)

    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )