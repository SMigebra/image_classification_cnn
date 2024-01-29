import streamlit as st 
import tensorflow as tf 
import cv2
from PIL import Image, ImageOps
import numpy as np

# st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model("F:/igebra/internship/ai ready/machine learning/image_classification_cnn/model.hdf5")
    return model

model = load_model()

st.title("CIFAR-10 Image Classification")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


def import_and_predict(image_data, model):
    size = (32, 32)
    image = np.array(image_data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    img_reshape = np.expand_dims(image, axis=0)
    prediction = model.predict(img_reshape)
    return prediction


if uploaded_file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    print(predictions)
    print(np.argmax(predictions))
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] 
    print(classes[np.argmax(predictions)])
    string = ("This image is most likely is :")
    st.success(f"This image most likely contains: {classes[np.argmax(predictions)]}")

