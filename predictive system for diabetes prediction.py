# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

# loading the saved model
#when using python, change the \ to / when writing path name
loaded_model = pickle.load(open("C:/machine learning/diabetes prediction system/trained_model.sav", 'rb'))

input_data = (10,115,0,0,0,35.3,0.134,29)
#changing the variable input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array as we are predicting for one instance.
#as we trained the model on many instances if we dont reshape then the model expects 768 data points.
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0] == 0):
  print("the person is non-diabetic")
else :
  print("the person is diabetic")
