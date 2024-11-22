import streamlit as st
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import tensorflow.keras.backend as K

# Define constants
BASE_DIR = '/root/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1'
WORKING_DIR = '/content/working'
MAX_LENGTH = 34  # Set your max length based on your data
Vocab_SIZE = 8525  # Set the vocab size based on tokenizer fit



# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to predict the caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Streamlit app
st.title('Image Captioning with VGG16 and LSTM')

st.markdown("Upload an image, and the model will generate a caption for it.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Process the image for prediction
    image_path = os.path.join(WORKING_DIR, 'temp_image.jpg')
    image.save(image_path)
    
    # Load and preprocess the image for feature extraction
    img = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(img)
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
    image_array = preprocess_input(image_array)

    # Extract features using VGG16 model
    feature = model.predict(image_array, verbose=0)
    
    # Generate a caption for the uploaded image
    caption = predict_caption(model, feature, tokenizer, MAX_LENGTH)
    
    # Display the generated caption
    st.subheader("Generated Caption:")
    st.write(caption)