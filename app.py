# Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page layout
st.set_page_config(layout='wide')

# Credits
with st.container():
    st.write('Author : Rutuja Patil')
    st.title('Laptops Prices Prediction\n')
    st.write('This prediction model is designed to predict the prices of laptops based on their features.')

# Load Cleaned data
df = pd.read_csv('laptop-clean.csv')

# Load Preprocessor
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Inputs
st.write('Laptop Brand & Type')
Company = st.selectbox('Company', df['Company'].unique())
TypeName = st.selectbox('Type', df['TypeName'].unique())
WeightKG = st.number_input('Weight (KG)')
TouchScreen = st.selectbox('Touch Screen', ['Yes', 'No'])
PanelType = st.selectbox('Panel Type', df['PanelType'].unique())
Resolution = st.selectbox('Resolution', df['Resolution'].unique())
Inches = st.number_input('Screen Size (Inches)')

st.write('Processor')
RamGB = st.number_input('RAM (GB)')
CpuBrand = st.selectbox('CPU Brand', df['CpuBrand'].unique())
GHz = st.number_input('CPU Speed (GHz)')
CpuVersion = st.selectbox('CPU Version', df['CpuVersion'].unique())

st.write('Storage')
MainMemory = st.number_input('Main Memory (GB)')
MainMemoryType = st.selectbox('Main Memory Type', df['MainMemoryType'].unique())
secMem = st.checkbox('Secondary Memory')
SecondMemory = st.number_input('Second Memory (GB)') if secMem else 0
SecondMemoryType = st.selectbox('Second Memory Type', df['SecondMemoryType'].unique()) if secMem else 'None'

st.write('Graphics Card')
GpuBrand = st.selectbox('GPU Brand', df['GpuBrand'].unique())
GpuVersion = st.selectbox('GPU Version', df['GpuVersion'].unique())

st.write('Operating System')
OpSys = st.selectbox('Operating System', df['OpSys'].unique())

# Prepare input data for prediction
new_data = {'Company': Company, 'TypeName': TypeName, 'Inches': Inches, 'RamGB': RamGB, 
        'OpSys': OpSys,
         'WeightKG': WeightKG, 'GHz': GHz, 'CpuBrand': CpuBrand, 'CpuVersion': CpuVersion, 
         'MainMemory': MainMemory, 'SecondMemory': SecondMemory,
         'MainMemoryType':MainMemoryType ,'SecondMemoryType':SecondMemoryType,
         'TouchScreen':TouchScreen,'Resolution':Resolution,
         'PanelType':PanelType , 'GpuBrand':GpuBrand,'GpuVersion':GpuVersion}
new_data = pd.DataFrame(new_data, index=[0])
new_data_preprocessed = preprocessor.transform(new_data)
st.write('- <p style="font-size:26px;"> Laptop Specs</p>',unsafe_allow_html=True)
new_data
# Prediction
log_price = model.predict(new_data_preprocessed) # in log scale
price = np.expm1(log_price) # in original scale
with st.container():
    coll1, coll2, coll3 = st.columns([3,6,1])

    with coll1:
            st.write("     ")

    with coll2:
                # Output
            if st.button('Predict'):
                 st.markdown('# Price in USD:')
                 price[0]

    with coll3:
            st.write("")
