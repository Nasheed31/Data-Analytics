#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import folium
from datetime import datetime
import warnings
from streamlit_folium  import st_folium
import streamlit as st
import altair as alt


# Load your crime data
crime_data = pd.read_csv(r'C:\Users\areeb\crime_train.csv')

# Data preprocessing and feature engineering
X = crime_data[['LATITUDE', 'Longitude', 'HOUR', 'DAY_OF_WEEK']]
y_hotspot = crime_data['crime_type']  # Assuming 'crime_type' is the target variable for crime hotspot
y_day_of_week = crime_data['DAY_OF_WEEK']  # Assuming 'DAY_OF_WEEK' is the target variable for day of the week

# Convert days of the week to numerical values using label encoding
day_of_week_mapping = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
X['DAY_OF_WEEK'] = X['DAY_OF_WEEK'].map(day_of_week_mapping)

# Split the data into training and testing sets
X_train, X_test, y_hotspot_train, y_hotspot_test, y_day_of_week_train, y_day_of_week_test = train_test_split(
    X, y_hotspot, y_day_of_week, test_size=0.2, random_state=42
)

# Model for predicting crime types (binary classification)
hotspot_model = RandomForestClassifier()
hotspot_model.fit(X_train, y_hotspot_train)

# Model for predicting day of the week (multiclass classification)
day_of_week_model = RandomForestClassifier()
day_of_week_model.fit(X_train, y_day_of_week_train)

# Make predictions
hotspot_predictions = hotspot_model.predict(X_test)
day_of_week_predictions = day_of_week_model.predict(X_test)

# Evaluate the models
hotspot_accuracy = accuracy_score(y_hotspot_test, hotspot_predictions)
hotspot_report = classification_report(y_hotspot_test, hotspot_predictions)

day_of_week_accuracy = accuracy_score(y_day_of_week_test, day_of_week_predictions)
day_of_week_report = classification_report(y_day_of_week_test, day_of_week_predictions)

# Print evaluation metrics
print(f'Crime Type Model Accuracy: {hotspot_accuracy}')
print('Crime Type Classification Report:\n', hotspot_report)

print(f'Day of the Week Model Accuracy: {day_of_week_accuracy}')
print('Day of the Week Classification Report:\n', day_of_week_report)

# Streamlit web app
st.header("PROJECT TITLE: CRIME GUARD")
st.write("In an era characterized by unprecedented technological advancements, the intersection of data science and law enforcement has emerged as a powerful tool to tackle and prevent criminal activities. The project at hand,CRIME GUARD endeavors to harness the vast reservoirs of crime data to develop a robust predictive model capable of foreseeing potential criminal incidents. By amalgamating cutting-edge data analytics techniques with comprehensive crime datasets, this project aims to empower law enforcement agencies with actionable insights, facilitating a proactive approach to public safety.")
st.button('[Click Here for Dataset]')
background_color = """
    <style>
        body {
            background-color: maroon; /* You can replace this with your desired color code */
        }
    </style>
"""
st.markdown(background_color, unsafe_allow_html=True)
st.title("Crime Prediction Map")

# User input for state, district, and crime type
state = st.selectbox("Select State", crime_data['STATE'].unique())
district = st.selectbox("Select District", crime_data[crime_data['STATE'] == state]['DISTRICT'].unique())
crime_type = st.selectbox("Select Crime Type", crime_data['crime_type'].unique())

# Filter data based on user input
filtered_data = crime_data[(crime_data['STATE'] == state) & (crime_data['DISTRICT'] == district) & (crime_data['crime_type'] == crime_type)]


# Create a map
#crime_map = folium.Map(location=[crime_data['LATITUDE'].mean(), crime_data['Longitude'].mean()], zoom_start=12)
crime_map = folium.Map(location=[filtered_data['LATITUDE'].mean(), filtered_data['Longitude'].mean()], zoom_start=12,width=800)


#for index, row in filtered_data.iterrows():
    #popup_text = f"Incident Number: {row['INCIDENT_NUMBER']}, Crime Type: {hotspot_model.predict([row[['LATITUDE', 'LONGITUDE', 'HOUR', 'DAY_OF_WEEK']]])[0]}, Day: {row['DAY_OF_WEEK']}, Time: {row['HOUR']}"
    #folium.Marker([row['LATITUDE'], row['LONGITUDE']], popup=popup_text).add_to(crime_map)

# Add markers for predicted crime hotspots
for index, row in X_test.iterrows():
    popup_text = f"Crime Type: {hotspot_model.predict([row])[0]}, Day: {row['DAY_OF_WEEK']}, Time: {row['HOUR']}"
    folium.Marker([row['LATITUDE'], row['Longitude']], popup=popup_text).add_to(crime_map)

# State-wise analysis chart
st.subheader("State-wise Analysis of Crime Types")
state_chart_data = crime_data.groupby(['STATE', 'crime_type']).size().reset_index(name='count')
state_chart = alt.Chart(state_chart_data).mark_bar().encode(
    x='STATE',
    y='count',
    color='crime_type',
    tooltip=['STATE', 'crime_type', 'count']
).interactive()

# Save the map as an HTML file or display it
crime_map.save('crime_predictions_map.html')
st_folium(crime_map)


# In[ ]:




