import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Directory for storing the trained model
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'house_price_model.pkl')

# Ensure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Preprocess the data
def preprocess_data(data):
    # We're predicting 'price' based on features 'area', 'bedrooms', 'bathrooms'
    data = data[['area', 'bedrooms', 'bathrooms', 'price']].dropna()
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model Mean Squared Error: {mse}')
    
    # Save the model in the 'model' directory
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    return model

# Load model for predictions
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

# Make predictions
def predict(area, bedrooms, bathrooms):
    model = load_model()
    prediction = model.predict([[area, bedrooms, bathrooms]])
    return prediction[0]

if __name__ == "__main__":
    data = load_data('D:/MLOPS-ASSIGNMENT-01/data/house_prices.csv')
    X, y = preprocess_data(data)
    train_model(X, y)
