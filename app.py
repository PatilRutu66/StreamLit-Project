# Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page layout
st.set_page_config(layout='wide')

# Credits
with st.container():
    st.write('Author : @RutujaPatil')
    st.write('LinkedIn : https://www.linkedin.com/in/rutujapatil06/')
    st.title('StreamLit Project - Laptop Prices Prediction\n')
    st.write('This prediction model is designed to forecast laptop costs according to their features. A more comprehensive analysis is available on my Github.')
    st.write('Feel free to contact me to receive the dataset & Python notebook.')

# Load Cleaned data
@st.cache  # Cache the data to prevent reloads on every interaction
def load_data():
    data = pd.read_csv('laptop-clean.csv')
    return data

df = load_data()

# Load Preprocessor and Model
@st.cache(allow_output_mutation=True)  # Cache to prevent reloads, allow_output_mutation for larger objects
def load_model_and_preprocessor():
    preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return preprocessor, model

preprocessor, model = load_model_and_preprocessor()

# Inputs
st.write('Laptop Brand & Type')
Company = st.selectbox('Company', df['Company'].unique())
# Include more inputs as necessary, ensure they are collected into a DataFrame for processing

# Collect inputs into DataFrame
# Example of input collection, ensure you collect all necessary inputs
input_data = {
    'Company': [Company],
    # Add other fields here
}

input_df = pd.DataFrame(input_data)

# Prediction
if st.button('Predict'):
    # Preprocess the new data using the loaded 'preprocessor'
    new_data_preprocessed = preprocessor.transform(input_df)
    
    # Use the preprocessed data to make a prediction with the loaded model
    log_price = model.predict(new_data_preprocessed)  # in log scale
    price = np.expm1(log_price)  # in original scale

    # Output the prediction to the user
    st.markdown('### Price in USD:')
    st.write(price[0])
