#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
# Load the model
df = pd.read_csv(r'C:\Users\areeb\Automobile_modified.csv')
# Select relevant features and target variable
# For simplicity, let's use only numerical features for this example
features = df[['symboling', 'wheel-base', 'length', 'width', 'height', 'curb-weight',
               'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower',
               'peak-rpm', 'city-mpg', 'highway-mpg']]
target = df['price']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Automobile Price Prediction")
st.sidebar.header("Input Parameters for Price")

# Input parameters
new_data = {
    'symboling': st.sidebar.slider('Symboling', -3, 3, 0),
    'wheel-base': st.sidebar.slider('Wheel Base', 80.0, 120.0, 80.2),
    'length': st.sidebar.slider('Length', 140, 250, 168),
    'width': st.sidebar.slider('Width', 50, 80, 71),
    'height': st.sidebar.slider('Height', 40, 80, 52),
    'curb-weight': st.sidebar.slider('Curb Weight', 1000, 5000, 1500),
    'engine-size': st.sidebar.slider('Engine Size', 50, 400, 150),
    'bore': st.sidebar.slider('Bore', 2.0, 5.0, 3.47),
    'stroke': st.sidebar.slider('Stroke', 2.0, 5.0, 2.50),
    'compression-ratio': st.sidebar.slider('Compression Ratio', 7.0, 25.0, 9.0),
    'horsepower': st.sidebar.slider('Horsepower', 50, 300, 150),
    'peak-rpm': st.sidebar.slider('Peak RPM', 4000, 7000, 5000),
    'city-mpg': st.sidebar.slider('City MPG', 10, 50, 19),
    'highway-mpg': st.sidebar.slider('Highway MPG', 10, 50, 27),
}

# Create a DataFrame from the input data
new_data_df = pd.DataFrame([new_data])

# Scale the input data using the same scaler
new_data_scaled = scaler.transform(new_data_df)

# Prediction
price_predictions = knn_model.predict(new_data_scaled)

# Display the predicted output
st.header("Predicted Output")
st.write(f"The predicted automobile price is: ${price_predictions[0]:,.2f}")

# Evaluate the model on the test set
y_test_pred = knn_model.predict(X_test_scaled)

# Display the model evaluation
#st.header("Model Evaluation")
#mse = mean_squared_error(y_test, y_test_pred)
#st.write(f"Mean Squared Error on Test Data: {mse}")


from sklearn.compose import ColumnTransformer
df=pd.read_csv(r"C:\Users\areeb\Downloads\archive (3)\Auto Sales data.csv")
#sns.pairplot(df, hue='PRODUCTLINE')
features=df[['QUANTITYORDERED', 'PRICEEACH', 'PRODUCTLINE', 'COUNTRY']]
target=df['SALES']
X_train, X_test, Y_train, Y_test=train_test_split=train_test_split(features, target, test_size=0.2, random_state=45)

# Define transformers for numerical and categorical columns
numeric_features = ['QUANTITYORDERED', 'PRICEEACH']
categorical_features = ['PRODUCTLINE', 'COUNTRY']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform the test data using the preprocessor fitted on the training data
X_test_scaled = preprocessor.transform(X_test)

knn_model=KNeighborsRegressor(n_neighbors=11)
knn_model.fit(X_train_scaled, Y_train)

new_data={
    'QUANTITYORDERED': 25, 'PRICEEACH':100, 'PRODUCTLINE':'Classic Cars', 'COUNTRY':'Australia'
}

new_data_df=pd.DataFrame([new_data])
new_data_scaled = preprocessor.transform(new_data_df)

predictions=knn_model.predict(new_data_scaled)

print(predictions)
# Check the shapes of Y_test and predictions
#print(Y_test.shape, predictions.shape)
#mse = mean_squared_error(Y_test, predictions)
#print(mse)

st.title('Automobile Sales Predictions')
st.sidebar.header("Input Parameter for Sales")
{  
    QUANTITYORDERED:= st.sidebar.slider('QUANTITYORDERED', 0, 100, 25), 
    PRICEEACH:= st.sidebar.slider('PRICEEACH', 0,100, 253),
    PRODUCTLINE:= st.sidebar.selectbox('PRODUCTLINE', df['PRODUCTLINE'].unique()),
    COUNTRY:= st.sidebar.selectbox('COUNTRY', df['COUNTRY'].unique())
}
user_input = {
    'QUANTITYORDERED':QUANTITYORDERED,
    'PRICEEACH': PRICEEACH,
    'PRODUCTLINE': PRODUCTLINE,
    'COUNTRY': COUNTRY
}

user_input_df=pd.DataFrame([user_input], index=[0])
scaled_input=preprocessor.transform(user_input_df)
sales_predictions=knn_model.predict(scaled_input)

st.subheader("Predicted Sales:")
st.write(sales_predictions[0])

# Visualize predicted sales using a bar chart
st.subheader("Predicted Sales and Price Chart:")
fig, ax = plt.subplots()
ax.plot(['sales_predictions'], [sales_predictions[0]], marker='o')
ax.plot(['price_predictions'], [price_predictions[0]], marker='o', label='Predicted Price')

ax.set_ylabel('Sales and Prices')
ax.set_xlabel('Prediction')
st.pyplot(fig)


# In[6]:





# In[ ]:




