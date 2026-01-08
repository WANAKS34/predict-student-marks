import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Marks Predictor",page_icon= "",layout="centered")

st.title("Student Mark Predictor")
st.write("Entre the number of hours Studied (1-18) And **Click Predict** To see The Predicted Marks")

#load the model
def load_model(model):
    with open(model,"rb") as f:
        slr = pickle.load(f)
    return slr 

try:
    model = load_model("lr.pkl")
except Exception as e: 
    st.error("Your Pickle file not found....")
    st.exception("Failed to lead The Model",e)
    st.stop()

hours = st.number_input("Hours_Studied",
                        min_value=1.0,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                        format= "%.1f")

if st.button("Predict"):
    try:
        X = np.array([[hours]])
        predictions = model.predict(X)
        predictions = predictions[0]

        st.success(f"Predicted Marks : {predictions:.1f}")
        st.write("Note : This is ML Model Prediction **Result MAy Vary**")
    except Exception as e: 
        st.error(f"prediction Failed : {e}")