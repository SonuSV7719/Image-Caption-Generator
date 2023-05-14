import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from keras.models import Model


# Load Models
model1 = keras.models.load_model('./Models/Image_Caption_Genreation_Model_3.h5')
# model2 = keras.models.load_model('./Models/Image_Caption_Genreation_Model_4.h5')
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load Captions.txt
with open('./Models/captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = ' '.join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            #Remove digits, special characters
            caption = caption.replace('[^A-Za-z]', '')
            #Remove Extra spaces
            caption = caption.replace('\s+', ' ')
            caption = 'nstartn ' + " ".join([word for word in caption.split() if len(word)>1]) + ' nendn'
            captions[i] = caption
clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(caption.split()) for caption in all_captions)

# UI Code
reshaped_img = ''
st.set_page_config(page_title='Image Caption Generator', page_icon='✍🏻')
st.title("Image Caption Generator")
uploaded_image = st.file_uploader(label='Upload Image.. Only support png, jpg & jpeg...', type=['png', 'jpg', 'jpeg'])
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    resized_img = cv2.resize(img, (224, 224))
    reshaped_img = resized_img.reshape((1, 224, 224, 3))
    st.image(resized_img, caption='Uploaded Image') 
else:
    st.write("No image file uploaded")

# To store output 
if not hasattr(st.session_state, 'output'):
    st.session_state.output = ""
# To store output 
# if not hasattr(st.session_state, 'output2'):
#     st.session_state.output = ""

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

## Generate Caption fo an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'nstartn'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'nendn':
            break
    return in_text

def extract_feature(model, image):
  feature = model.predict(image)
  return feature

def generate_caption(image_feature, model):
    y_pred = predict_caption(model, image_feature, tokenizer, max_length)
    for i in ['nstartn', 'nendn']:
        final_output = y_pred.replace(i, '')
    st.session_state.output = f"------------------------------Predicted-------------------------<br>{final_output}"


def predict():
    if uploaded_image is not None:
        generate_caption(extract_feature(vgg_model, reshaped_img), model1)
        generate_caption(extract_feature(vgg_model, reshaped_img), model2)
    else:
        st.write("Please Upload image")

st.markdown(st.session_state.output, unsafe_allow_html=True)
button = st.button(label="Generate", on_click=predict)


