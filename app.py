import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the trained machine learning model
model = load_model('my_model.h5')

# Load the scaler object used during training
scaler_file = 'scaler.pkl'
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# Define the function for making predictions
def predict_outcome(age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
                    genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker,
                    chest_pain, coughing_blood, fatigue, weight_loss, shortness_of_breath, wheezing,
                    swallowing_difficulty, clubbing, frequent_cold, dry_cough, snoring):
    
    gender_mapping = {'Male': 1, 'Female': 2}
    gender_numeric = gender_mapping.get(gender, 0)

    data = np.array([age, gender_numeric, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
                     genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker,
                     chest_pain, coughing_blood, fatigue, weight_loss, shortness_of_breath, wheezing,
                     swallowing_difficulty, clubbing, frequent_cold, dry_cough, snoring]).reshape(1, -1)

    # Standardize the data
    new_data_scaled = scaler.transform(data)

    # Make predictions
    predictions = model.predict(new_data_scaled)

    # Convert the predictions to class labels
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Determine the predicted outcome based on the prediction
    if predicted_class == 0:
        outcome = 'Low'
    elif predicted_class == 1:
        outcome = 'Medium'
    else:
        outcome = 'High'

    return outcome

# Streamlit app begins here
st.title('Lung Cancer Risk Prediction')

# Add form elements for user input
age = st.number_input('Age', min_value=0, max_value=150, step=1)
gender = st.selectbox('Gender', ['Male', 'Female'])
air_pollution = st.number_input('Air Pollution', min_value=0, max_value=10, step=1)
alcohol_use = st.number_input('Alcohol Use', min_value=0, max_value=10, step=1)
dust_allergy = st.number_input('Dust Allergy', min_value=0, max_value=10, step=1)
occupational_hazards = st.number_input('Occupational Hazards', min_value=0, max_value=10, step=1)
genetic_risk = st.number_input('Genetic Risk', min_value=0, max_value=10, step=1)
chronic_lung_disease = st.number_input('Chronic Lung Disease', min_value=0, max_value=10, step=1)
balanced_diet = st.number_input('Balanced Diet', min_value=0, max_value=10, step=1)
obesity = st.number_input('Obesity', min_value=0, max_value=10, step=1)
smoking = st.number_input('Smoking', min_value=0, max_value=10, step=1)
passive_smoker = st.number_input('Passive Smoker', min_value=0, max_value=10, step=1)
chest_pain = st.number_input('Chest Pain', min_value=0, max_value=10, step=1)
coughing_blood = st.number_input('Coughing Blood', min_value=0, max_value=10, step=1)
fatigue = st.number_input('Fatigue', min_value=0, max_value=10, step=1)
weight_loss = st.number_input('Weight Loss', min_value=0, max_value=10, step=1)
shortness_of_breath = st.number_input('Shortness of Breath', min_value=0, max_value=10, step=1)
wheezing = st.number_input('Wheezing', min_value=0, max_value=10, step=1)
swallowing_difficulty = st.number_input('Swallowing Difficulty', min_value=0, max_value=10, step=1)
clubbing = st.number_input('Clubbing', min_value=0, max_value=10, step=1)
frequent_cold = st.number_input('Frequent Cold', min_value=0, max_value=10, step=1)
dry_cough = st.number_input('Dry Cough', min_value=0, max_value=10, step=1)
snoring = st.number_input('Snoring', min_value=0, max_value=10, step=1)

# Add a button to trigger the prediction
if st.button('Predict', key='predict_button', help='Click to make a prediction'):
    outcome = predict_outcome(age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
                              genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking, passive_smoker,
                              chest_pain, coughing_blood, fatigue, weight_loss, shortness_of_breath, wheezing,
                              swallowing_difficulty, clubbing, frequent_cold, dry_cough, snoring)
    st.success(f'Predicted Risk Level: {outcome}')
