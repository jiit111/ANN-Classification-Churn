import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

# Load the trained model
model= tf.keras.models.load_model('model.h5')

# Load the encoders and scaler

with open('one_hot_encoder_geography.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##streamlit app
st.title('customer churn prediction')

geography= st.selectbox('Geography',label_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoder = label_encoder_geo.transform([[geography]]).toarray()
onehot_columns = label_encoder_geo.get_feature_names_out(['Geography'])
geo_encoded_df=pd.DataFrame(geo_encoder,columns=onehot_columns)
# Concatenate the one-hot encoded geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled=scaler.transform(input_data)

##predict
prediction=model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display the prediction result
if prediction_proba > 0.5:
    st.write(f"Prediction: Customer is likely to churn (Probability: {prediction_proba:.2f})")  
else:
    st.write(f"Prediction: Customer is likely to stay (Probability: {1 - prediction_proba:.2f})")