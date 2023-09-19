import streamlit as st
import numpy as np
import pickle
import pandas as pd


model = pickle.load(open('xgmodel.pkl', 'rb'))

def main():
    st.title('Rem')

    #input
    Year = st.text_input("Year")
    remittance_recieved= st.text_input('remittance_recieved')
    country_name= st.text_input('country_name')
    new_income_group= st.text_input('new_income_group')
    new_region= st.text_input('new_region')

    if st.button('Prediction'):
        makeprediction = model.predict([[Year, remittance_recieved, country_name, new_income_group, new_region]])
        output = round(makeprediction[0],2)
        st.success('GDP is {}'.format(output))

if __name__ == "__main__":
    main()