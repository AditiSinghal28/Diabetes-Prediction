# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 23:26:18 2025

@author: aditi
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
#when using python, change the \ to / when writing path name
loaded_model = pickle.load(open("C:/machine learning/diabetes prediction system/trained_model.sav", 'rb'))


#creating a function for prediction

def diabetes_prediction(input_data) :
    
    #changing the variable input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping the array as we are predicting for one instance.
    #as we trained the model on many instances if we dont reshape then the model expects 768 data points.
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0] == 0):
      return "the person is non-diabetic"
    else :
      return"the person is diabetic"
      
      
def main():
    #giving a title to the web page
    st.title("Diabetes Prediction Web App")
    
    #getting input data from user
    
    
    Pregnancies = st.text_input('Number of Pregnancies : ')
    Glucose = st.text_input('Glucose Level : ')
    BloodPressure = st.text_input('Blood Pressure Value : ')
    SkinThickness = st.text_input('Skin Thickness : ')
    Insulin = st.text_input('Insulin Level : ')
    BMI = st.text_input('BMI : ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function : ')
    Age = st.text_input('Age of the person : ')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)
    


if __name__ == '__main__' :
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    