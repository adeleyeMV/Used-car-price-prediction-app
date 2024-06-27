import pandas as pd
import numpy as np
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import os


# Load model and preprocessing artifacts
script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, 'car_price_prediction.joblib')

# Load the file using the absolute path
artifact = joblib.load(file_path)

# Function to format price
def format_price(price):
    formatted_price = "{:,.2f}".format(price)
    return f'₦{formatted_price}'

# Function to format engine size
def format_engine_size(engine_size):
    return f'{engine_size}cc'

# Function to format mileage
def format_mileage(mileage):
    return f'{mileage:.2f}km'

# Function to check missing inputs
def check_missing_inputs(data):
    return [key for key, value in data.items() if value == '']

# Function to make predictions
def predict(data):
    df = pd.DataFrame([data])
    X_data = pd.DataFrame(artifact['preprocessing'].transform(df), columns=artifact['preprocessing'].get_feature_names_out())

    # Display loading indicator
    with st.spinner("Predicting..."):
        prediction = artifact['model'].predict(X_data)

    estimated_price = np.array(np.expm1(prediction))
    formatted_price = format_price(estimated_price[0])

    return X_data, formatted_price  # Return X_data along with the price

# Data Input Page
def input_page():
    st.title('Used Car Price Prediction')

    make = st.selectbox("Car Make", ['', 'Honda', 'Acura', 'Peugeot', 'Nissan', 'Kia', 'Hyundai', 'Toyota',
        'Ford', 'Lexus', 'Mazda', 'Volkswagen', 'Mercedes-Benz',
        'Land Rover', 'Dodge', 'Subaru', 'Mitsubishi', 'Infiniti', 
        'Volvo', 'Audi', 'Porsche', 'Pontiac', 'BMW', 'Jeep',
        'Chevrolet', 'Suzuki', 'Hummer', 'Iveko'])

    model = st.text_input("Model", ).title()
    y_of_manu = st.selectbox("Year Of Manufacture", ['',2000,2001,2002,2003,2004,2005,2006,2007, 2008,
                                                    2009,2010,2011,2012,2013,2014,2015,2016,2017,
                                                    2018,2019,2020,2021,2022, 2023])

    color = st.text_input("Color", ).title()
    condition = st.selectbox("Condition", ['','Nigerian Used', 'Foreign Used'])
    mileage = st.number_input("Mileage (km)", min_value=0.0)
    eng_size = st.number_input("Engine Size (cc)", min_value=0.0)
    bought_condition = st.selectbox("Bought Condition", ['','Imported', 'Registered', 'Brand new'])
    registered_state = st.selectbox("Registered State", ['', 'Abia', 'Abuja','Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 'Benue', 'Borno',
                                                        'Cross River', 'Delta', 'Ebonyi', 'Edo', 'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa',
                                                        'Kaduna', 'Kano', 'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nasarawa', 'Niger',
                                                        'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers', 'Sokoto', 'Taraba', 'Yobe', 'Zamfara']).lower()

    user_input = {
        'Make': make,
        'Model': model,
        'Year of manufacture': y_of_manu,
        'Colour': color,
        'Condition': condition,
        'Mileage': mileage,
        'Engine Size': eng_size,
        'Bought Condition': bought_condition,
        'Registered state': registered_state
    }

    # ...

    def display_details(user_input):
        details = [
            f'{key}: {format_mileage(value)}' if key == 'Mileage' else
            f'{key}: {format_engine_size(value)}' if key == 'Engine Size' else
            f'{key}: {value}'
            for key, value in user_input.items()
        ]
        return f'Details: {", ".join(details)}'

    # Button to make prediction
    if st.button('Predict Price'):
        missing_fields = check_missing_inputs(user_input)
        if missing_fields:
            st.error(f"Please fill in the following fields: {', '.join(missing_fields)}")
        else:
            X_data, result = predict(user_input)
            st.success(result)
            st.write(display_details(user_input))
            st.write(f'Estimated Price: {format_price(np.expm1(artifact["model"].predict(X_data))[0])}')
