import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import time


#covid model
@st.cache(allow_output_mutation=True)
def load_modelcovid():
    modelcovid = tf.keras.models.load_model('C:/Users/fahad/Downloads/modelCovid.h5')
    return modelcovid

def import_and_predictcovid(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

with st.spinner('Model is being loaded..'):
    modelcovid = load_modelcovid()

st.write("""
         #  Detection of Lung Diseases
         #  Covid
         """

         )

filecovid = st.file_uploader("Please upload an Xray scan file", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)


if filecovid is None:
   st.text("Please upload an image file")
else:
    image = Image.open(filecovid)
    st.image(image, use_column_width=True)
    predictions = import_and_predictcovid(image, modelcovid)
    if st.button("Predict",key="covidbutton"):
        time.sleep(5)
        if predictions == 0:
            string = "This is Covid"
        else:
            string = " This is Normal"
        st.success(string)

#Pneumonia model

st.write("""
         #  Pneumonia
         """

         )


@st.cache(allow_output_mutation=True)
def load_modelpneumonia():
    return tf.keras.models.load_model("C:/Users/fahad/Downloads/pneumonia/KerasDF/model/modelPneumonia.h5")


def make_predictionpneumonia(uploaded_image, model):
    uploaded_image.save("out.jpg")
    image = cv2.imread("out.jpg")
    image = cv2.resize(image, dsize=(200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    image = image.reshape((1, image.shape[0], image.shape[1]))
    prediction = np.argmax(model.predict(image))
    return prediction

filepneumonia = st.file_uploader("Please upload an Xray scan file", type=["jpg", "png"],key= "Pneumonia")
st.set_option('deprecation.showfileUploaderEncoding', False)

modelpneumonia = load_modelpneumonia()
if filepneumonia is None:
   st.text("Please upload an image file")
else:
    image1 = Image.open(filepneumonia)
    st.image(image1, use_column_width=True)
    predictions = make_predictionpneumonia(image1, modelpneumonia)
    if st.button("Predict", key= "Pnemoniabutton"):
        time.sleep(5)
        if predictions == 0:
            string = "This is Pneumonia"
        else:
            string = " This is Normal"
        st.success(string)

#Effusion model

st.write("""
         #  Effusion
         """

         )


@st.cache(allow_output_mutation=True)
def load_modeleffusion():
    return tf.keras.models.load_model("C:/Users/fahad/Downloads/modelEffusion1.hdf5")


def make_predictioneffusion(uploaded_image, model):
    uploaded_image.save("out.jpg")
    image = cv2.imread("out.jpg")
    image = cv2.resize(image, dsize=(256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image)
    image = image.reshape((1, image.shape[0], image.shape[1]))
    prediction = np.argmax(model.predict(image))
    return prediction

fileeffusion = st.file_uploader("Please upload an Xray scan file", type=["jpg", "png"],key= "Effusion")
st.set_option('deprecation.showfileUploaderEncoding', False)

modeleffusion = load_modeleffusion()
if fileeffusion is None:
   st.text("Please upload an image file")
else:
    image2 = Image.open(fileeffusion)
    st.image(image2, use_column_width=True)
    predictions = make_predictioneffusion(image2, modeleffusion)
    if st.button("Predict", key= "Effusionbutton"):
        if predictions == 0:
            string = "This is Effusion"
        else:
            string = " This is Normal"
        st.success(string)




