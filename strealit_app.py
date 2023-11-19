import numpy as np
from PIL import Image
import streamlit as st
from train_fast_v1 import *

st.title("বাংলা নামতা using CNN from scratch")
st.markdown("Please upload an image file of a **Bengali Digit** ... [***preferably written in black on white background for optimal performance***]")


def predict_from_image(img):
    model = CNNModel()
    model.load_model_weights_pickle("model_weights_kaggle.pkl")

    # img = cv2.imread(url, cv2.IMREAD_COLOR)
    # print(img.shape)
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
    # st.write("")
    # st.write("Classifying...")
    pred, pred_distribution = predict_from_image(img)
    print(pred_distribution)
    ## show prediction distribution
    st.write("Prediction Distribution: ")
    st.bar_chart(pred_distribution)
    # st.write("Predicted: ", pred)
    # write predicted class in header 3
    st.markdown(f'## Predicted: `{pred}`')

## load a png image from url
url = "dataset/NumtaDB_with_aug/training-a/a00000.png"
## center the image

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.write(' ')

# with col2:
#     img = st.image(url, width=200)

# with col3:
#     st.write(' ')



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


# pred = predict_from_image(img)
# pred = predict(url)
# st.write("Predicted: ", pred)
