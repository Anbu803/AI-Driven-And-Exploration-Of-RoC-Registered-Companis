# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load your RoC registration data (replace 'data.csv' with your dataset)
data = pd.read_csv('data.csv')

# Data preprocessing
# Assume your dataset has columns 'Year' and 'Registration_Count'
# You may need to perform more extensive preprocessing depending on your data

# Split the data into training and testing sets
X = data['Year'].values.reshape(-1, 1)
y = data['Registration_Count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the model
y_pred = model.predict(X_test)

# Visualize the actual vs. predicted registration trends
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Registration Count')
plt.legend()
plt.title('Company Registration Trends Prediction')
plt.show()

# You can use this trained model to make future predictions as well
future_years = np.array([2024, 2025, 2026]).reshape(-1, 1)
future_predictions = model.predict(future_years)
print("Future Predictions for 2024, 2025, and 2026:")
for year, prediction in zip(future_years.flatten(), future_predictions):
    print(f"Year {year}: Predicted Count = {prediction:.2f}")
