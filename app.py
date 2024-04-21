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
# Ensure you have uploaded the 'laptop-clean.csv' file on the deployment platform
df = pd.read_csv('laptop-clean.csv')

# Load Preprocessor
# Ensure you have uploaded the 'preprocessor.pkl' file on the deployment platform
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Inputs

# Brand & Type
st.write('Laptop Brand & Type')
Company = st.selectbox('Company', df['Company'].unique())
# ...

# The rest of your Streamlit widgets and functionality here

# Prediction
# When the user clicks the 'Predict' button, the app will process the inputs and display the prediction
if st.button('Predict'):
    # Preprocess the new data using the loaded 'preprocessor'
    new_data_preprocessed = preprocessor.transform(new_data)
    
    # Use the preprocessed data to make a prediction with the loaded model
    log_price = model.predict(new_data_preprocessed)  # in log scale
    price = np.expm1(log_price)  # in original scale

    # Output the prediction to the user
    st.markdown('### Price in USD:')
    st.write(price[0])
