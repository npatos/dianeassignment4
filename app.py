import streamlit as st
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


st.write(""" 
    # Spam Classifier Web App

    This app predicts whether an email or text is considered ***spam (1) or ham (0)*** 
""")

# Get user input
user_input = st.text_input("Enter an email or text:")

st.write("***User Email: ***:" , user_input)

# Load the tokenizer object
tokenizer_file = "tokenize.sav"
tokenizer = pickle.load(open(tokenizer_file, "rb"))

# Prepare user input
user_input = [user_input.split(" ")]
text_seq = tokenizer.texts_to_sequences(user_input)
padded_text_seq = pad_sequences(text_seq, maxlen=4, padding="post") 

# Load the model (keras)
model_file = "model.h5"
bilstm_model = load_model(model_file, compile = False)

y_pred = bilstm_model.predict(padded_text_seq)
y_pred = np.argmax(y_pred, axis=1)

if st.button("Predict"):
    if y_pred[0] == 0:
        st.write("Prediction: ***Ham***")
    elif y_pred[0] == 1:
        st.write("Prediction: ***Spam***")
