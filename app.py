import cv2
import numpy as np
from PIL import Image
import streamlit as st
from src.model import CNNModel

st.title("CNN from scratch tested on বাংলা নামতা (Bangla Digit) Dataset")
st.markdown("Please upload an image file of a **Bengali Digit** ...")


def predict_from_image(img):
    model = CNNModel()
    model.load_model_weights_pickle("src/model_weights_kaggle.pkl")

    img = cv2.resize(img, (28, 28))
    print(img.shape)
    img = np.array(img)

    img = img.transpose(2, 0, 1)
    print(img.shape)

    img = img / 255.0
    img = img.reshape(1, 3, 28, 28)
    print(img.shape)

    img = 1 - img

    pred = model.forward(img)
    fina_pred = np.argmax(pred, axis=1)
    print(fina_pred)
    return fina_pred[0], pred[0]


## file uploader
uploaded_file = st.file_uploader("Choose a image file", type="png")
print(uploaded_file)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)
    img = np.array(image)
    print("SHAPE: ",img.shape)

    pred, pred_distribution = predict_from_image(img)
    print(pred_distribution)

    st.write("Prediction Distribution: ")
    st.bar_chart(pred_distribution)

    st.markdown(f'## Predicted: `{pred}`')


url = "dataset/NumtaDB_with_aug/training-a/a00000.png"


def predict(url):
    model = CNNModel()
    model.load_model_weights_pickle("model_weights_kaggle.pkl")

    img = cv2.imread(url, cv2.IMREAD_COLOR)
    print(img.shape)
    img = cv2.resize(img, (28, 28))
    print(img.shape)
    img = np.array(img)

    img = img.transpose(2, 0, 1)
    print(img.shape)

    img = img / 255.0
    img = img.reshape(1, 3, 28, 28)
    print(img.shape)

    img = 1 - img

    pred = model.forward(img)
    pred = np.argmax(pred, axis=1)
    print(pred)
    return pred[0]