import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

def prediction(flow_duration,fwd_IAT_total,Bwd_IAT_total, Fwd_packet_length, Init_win_bytes_fwd):
    prediction = classifier.predict([[flow_duration,fwd_IAT_total,Bwd_IAT_total, Fwd_packet_length, Init_win_bytes_fwd]])
    return prediction

filename = 'data.pkl'
with open(filename, 'rb') as file: 
  classifier = joblib.load(file)

st.title("Supervised Network Classification App")
st.write('This classification using **_Random Forest Classifier_**.')
st.write(pd.DataFrame({
    '':['AMAZON','IP_ICMP'], 
    ' ':['APPLE_ICLOUD', 'MS_ONE_DRIVE'],
    '   ':['CONTENT_FLASH', 'OFFICE_365'], 
    '    ':['HTTP_CONNECT', 'WHATSAPP'],
    'application':['INSTAGRAM','YAHOO'],
    }))
# Input Side
st.sidebar.header('User Input Parameters')

flow_duration = st.sidebar.number_input("Flow Duration")
fwd_IAT_total = st.sidebar.number_input("Forward Inter Arrival Time Total")
Bwd_IAT_total = st.sidebar.number_input("Backward Inter Arrival Time Total")
Fwd_packet_length = st.sidebar.number_input("Forward Packet Length Std")
Init_win_bytes_fwd = st.sidebar.number_input("Number Bytes in Initial Window Forward")
result = ""

if st.button("Classify"):
    result = prediction(flow_duration,fwd_IAT_total,Bwd_IAT_total, Fwd_packet_length, Init_win_bytes_fwd)
    with st.spinner('Classifying...'):
        time.sleep(3)
    m = classifier.predict_proba([[flow_duration,fwd_IAT_total,Bwd_IAT_total, Fwd_packet_length, Init_win_bytes_fwd]])
    
    if max(m[0])<=0.6 :
        st.success('Your Input is **NOT in this APPs**')
    else :
        app = np.where(m[0]==max(m[0]))
        n = m[0][app[0][0]]*100
        st.success('Your Input is **{0}** with accuracy {1:.2f}%'.format(result[0],n))
        st.balloons()