# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# Page layout
st.set_page_config(layout='wide')

# Image
image = Image.open('Picture.jpg')

# Credits
with st.container():
    st.write('Author: @Rutuja Patil')
    st.title('StreamLit application for Laptop Price Predictions\n')
    st.write('- <p style="font-size:26px;">This prediction model estimates laptop prices based on their features and specifications. For a more comprehensive analysis.</p>', unsafe_allow_html=True)
    coll1, coll2, coll3 = st.columns([3, 6, 1])

    with coll1:
        st.write("     ")

    with coll2:
        st.image(image, width=800)

    with coll3:
        st.write("")

# Load Cleaned data
df = pd.read_csv('laptop-clean.csv')

# Load Preprocessor and Model
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Inputs
st.write('- <p style="font-size:26px;">Laptop Brand & Type</p>', unsafe_allow_html=True)
Company = st.selectbox('Company', df['Company'].unique())
TypeName = st.selectbox('Type', df[df['Company'] == Company]['TypeName'].unique())
WeightKG = st.selectbox('Weight (KG)', np.sort(df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['WeightKG'].unique()))

# Screen Inputs
st.write('- <p style="font-size:26px;">Screen Specs</p>', unsafe_allow_html=True)
TouchScreen = st.selectbox('Touch Screen', ['Yes', 'No'])
TouchScreen = 1 if TouchScreen == 'Yes' else 0
PanelType = st.selectbox('Panel Type', df['PanelType'].unique())
Resolution = st.selectbox('Resolution', df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['Resolution'].unique())
Inches = st.number_input('Inches', float(df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['Inches'].min()), float(df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['Inches'].max()))

# Processor & RAM
st.write('- <p style="font-size:26px;">Processor & RAM</p>', unsafe_allow_html=True)
RamGB = st.selectbox('RAM (GB)', df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['RamGB'].unique())
CpuBrand = st.selectbox('CPU Brand', df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['CpuBrand'].unique())
GHz = st.selectbox('CPU GHz', df[(df['Company'] == Company) & (df['TypeName'] == TypeName) & (df['CpuBrand'] == CpuBrand)]['GHz'].unique())
CpuVersion = st.selectbox('CPU Version', df[(df['Company'] == Company) & (df['TypeName'] == TypeName) & (df['CpuBrand'] == CpuBrand) & (df['GHz'] == GHz)]['CpuVersion'].unique())

# Hard Disk
st.write('- <p style="font-size:26px;">Hard Disk Capacity</p>', unsafe_allow_html=True)
MainMemory = st.selectbox('Main Memory (GB)', df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['MainMemory'].unique())
MainMemoryType = st.selectbox('Main Memory Type', df[(df['Company'] == Company) & (df['MainMemory'] == MainMemory)]['MainMemoryType'].unique())
secMem = st.checkbox('Extra Hard Drive')
SecondMemory = 0.0
SecondMemoryType = 'None'
if secMem:
    SecondMemory = st.selectbox('Second Memory (GB)', df['SecondMemory'].unique()[1:])
    SecondMemoryType = st.selectbox('Second Memory Type', df['SecondMemoryType'].unique()[1:])

# Graphics Card
st.write('- <p style="font-size:26px;">Graphics Card</p>', unsafe_allow_html=True)
GpuBrand = st.selectbox('GPU Brand', df[(df['Company'] == Company) & (df['TypeName'] == TypeName)]['GpuBrand'].unique())
GpuVersion = st.selectbox('GPU Version', df[(df['Company'] == Company) & (df['TypeName'] == TypeName) & (df['GpuBrand'] == GpuBrand)]['GpuVersion'].unique())

# Operating System
st.write('- <p style="font-size:22px;">Operating System</p>', unsafe_allow_html=True)
OpSys = st.selectbox('Operating System', df[(df['Company'] == Company)]['OpSys'].unique())

# Preprocessing
new_data = {'Company': Company, 'TypeName': TypeName, 'Inches': Inches, 'RamGB': RamGB,
            'OpSys': OpSys, 'WeightKG': WeightKG, 'GHz': GHz, 'CpuBrand': CpuBrand,
            'CpuVersion': CpuVersion, 'MainMemory': MainMemory, 'SecondMemory': SecondMemory,
            'MainMemoryType': MainMemoryType, 'SecondMemoryType': SecondMemoryType,
            'TouchScreen': TouchScreen, 'Resolution': Resolution, 'PanelType': PanelType,
            'GpuBrand': GpuBrand, 'GpuVersion': GpuVersion}
new_data = pd.DataFrame(new_data, index=[0])

# Prediction
new_data_preprocessed = preprocessor.transform(new_data)
log_price = model.predict(new_data_preprocessed)  # in log scale
price = np.expm1(log_price)  # in original scale

# Display Prediction
if st.button('Predict'):
    st.markdown(f'# Price in USD: {price[0]:,.2f}')
