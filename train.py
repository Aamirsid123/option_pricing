from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle

# Load the data
data = pd.read_csv('train.csv')

# Preprocessing
X = data.drop(columns=['Id', 'OptionPrice'])  # Features
y = data['OptionPrice']  # Target

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['OptionType'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = make_pipeline(StandardScaler(), RandomForestRegressor())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"R^2 score: {score}")

from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse}, MAE: {mae}")

# Save the model to a file
with open('option_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)