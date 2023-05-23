import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from PIL import Image
# Loading Our final trained Knn model
model = open("../RandomForestClassifier.pkl", "rb")
rf = joblib.load(model)
st.title("HEART DESEASE PREDICTION PROJECT APP")

BMI = st.number_input('BMI', step=1.0, format="%2.2f")
Smoking = st.selectbox('Smoking', ['Yes', 'No'])
AlcoholDrinking = st.selectbox('AlcoholDrinking', ['Yes', 'No'])
Stroke = st.selectbox('Stroke', ['Yes', 'No'])
PhysicalHealth = st.slider('PhysicalHealth', min_value=0.0, max_value=30.0, step=1.0, format="%.1f")
MentalHealth = st.slider('MentalHealth', min_value=0.0, max_value=30.0, step=1.0, format="%.1f")
DiffWalking = st.selectbox('DiffWalking', ['Yes', 'No'])
Sex = st.selectbox('Sex', ['Male', 'Female'])
age_range = ['55-59', '80 or older', '65-69', '75-79',
             '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29']
AgeCategory = st.selectbox('AgeCategory', age_range)
Diabetic = st.selectbox('Diabetic', ['Yes', 'No'])
PhysicalActivity = st.selectbox('PhysicalActivity', ['Yes', 'No'])
gen_health_range = ['Very good', 'Fair', 'Good', 'Poor', 'Excellent']
GenHealth = st.selectbox('GenHealth', gen_health_range)
SleepTime = st.slider('SleepTime', min_value=0.0, max_value=24.0, step=1.0, format="%.1f")
Asthma = st.selectbox('Asthma', ['Yes', 'No'])
KidneyDisease = st.selectbox('KidneyDisease', ['Yes', 'No'])

age_label = LabelEncoder().fit(age_range)
gen_label = LabelEncoder().fit(gen_health_range)

if st.button('Predict'):
    AgeCategoryVal = int(age_label.transform([AgeCategory])[0])
    GenHealthVal = int(gen_label.transform([GenHealth])[0])
    # Create a DataFrame with user input
    data1 = {'BMI': float(BMI), 'Smoking': Smoking, 'AlcoholDrinking': AlcoholDrinking, 'Stroke': Stroke,
             'PhysicalHealth': PhysicalHealth, 'MentalHealth': MentalHealth, 'DiffWalking': DiffWalking, 'Sex': Sex,
             'AgeCategory': AgeCategoryVal, 'Diabetic': Diabetic, 'PhysicalActivity': PhysicalActivity,
             'GenHealth': GenHealthVal, 'SleepTime': SleepTime, 'Asthma': Asthma, 'KidneyDisease': KidneyDisease}
    st.write(data1)
    input_df = pd.DataFrame(data1, index=[0])
    # Replace values
    input_df = input_df[input_df.columns].replace(
        {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'No, borderline diabetes': 0, 'Yes (during pregnancy)': 1})
    st.write(input_df)

    # Make prediction
    prediction = rf.predict(input_df)

    # Display the prediction
    if prediction[0] == 0:
        st.write('The patient is unlikely to have heart disease.')
    else:
        st.write('The patient is likely to have heart disease.')
