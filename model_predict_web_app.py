import os

import streamlit as st
from dotenv import load_dotenv
import numpy as np
import pickle

loaded_model = pickle.load(open('trained_model3.sav', 'rb'))
# Configure Streamlit page settings
st.set_page_config(
    page_title="ChatBot powered by Gemini-Pro",
    layout="wide",  # Page layout option
)

# creating function for prediction.
def mice_class_predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data[:-1])
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # giving title for the web page.
    st.title("mice_class_predict_web_app")

    # getting input data from the user.
    input_variable_names = ['DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N',
                             'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N',
                             'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N',
                             'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N',
                             'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N',
                             'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N',
                             'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N',
                             'Genotype', 'Treatment', 'Behavior']

    # Collect user inputs in a list
    input_values = [float(st.text_input(f"Provide input for {var}", "0.0")) for var in input_variable_names]

    # code for prediction.
    # creating a button for the user.
    if st.button('mice class'):
        # Add 'class' as the last element in the input_values list
        input_values.append('class')
        # Prediction
        prediction = mice_class_predict(input_values)
        st.success(f'Predicted Class: {prediction[0]}')

if __name__ == '__main__':
    main()
