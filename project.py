# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime

# Load the dataset
df = pd.read_csv('coin_Aave.csv')

# Ensure data is loaded correctly
print(df.head())

# Plot the original data
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('coin Close Price History')
plt.legend()
plt.show()

# Data preprocessing
data = df.filter(['Close']).values   # Extract the 'Close' prices as a NumPy array.
scaler = MinMaxScaler(feature_range=(0, 1))    
scaled_data = scaler.fit_transform(data)   # Scale the data between 0 and 1.

# Define a function to create the training dataset
def create_dataset(dataset, time_step):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):    
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define the time step for the LSTM model
time_step = 60   # Use the last 60 data points as input for each prediction.
X, y = create_dataset(scaled_data, time_step)

# Reshape data for LSTM model
X = torch.tensor(X.reshape(X.shape[0], X.shape[1], 1), dtype=torch.float32)
y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# Splitting the data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model with corrected init method
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1


# Instantiate the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')


# Transform data back to original scale
train_predict = model(X_train).detach().numpy()
test_predict = test_outputs.detach().numpy()
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Prepare data for plotting
train_plot = np.empty_like(data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_plot = np.empty_like(data)
test_plot[:, :] = np.nan


# Calculate the start index for the test plot
test_start_index = len(train_predict) + time_step

# Adjust slice range based on test predictions and available data
test_end_index = test_start_index + len(test_predict)
if test_end_index > len(data):
    test_end_index = len(data)

# Place test predictions in the correct range
test_plot[test_start_index:test_end_index, :] = test_predict[:test_end_index - test_start_index]

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Training Predictions')
plt.plot(test_plot, label='Testing Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('coin Price Prediction using LSTM')
plt.legend()
plt.show()


# Forecast Future Values
future_steps = 60  # Number of future days to predict
model.eval()
predicted_values = []
last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)  # Get the last 60 days as input

# Predict future values
for _ in range(future_steps):
    with torch.no_grad():
        future_predict = model(torch.tensor(last_sequence, dtype=torch.float32))
    predicted_values.append(future_predict.item())  # Append predicted value to the list
    last_sequence = np.append(last_sequence[:, 1:, :], future_predict.detach().numpy().reshape(1, 1, 1), axis=1)
    

# Inverse transform the predictions
predicted_values = np.array(predicted_values).reshape(-1, 1)
predicted_values = scaler.inverse_transform(predicted_values)


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Training Predictions')
plt.plot(test_plot, label='Testing Predictions')
plt.plot(np.arange(len(scaled_data), len(scaled_data) + future_steps), predicted_values, label='Future Predictions', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('coin Price Prediction using LSTM')
plt.legend()
plt.show()

# Create a DataFrame for the future predictions
# Assuming the dataset has a 'Date' column in string format that needs to be extended
if 'Date' in df.columns:
    # Convert the last date in the dataset to a datetime object
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, future_steps + 1)]
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predicted_values.flatten()})
else:
    # If no 'Date' column is present, just create an index for future predictions
    future_df = pd.DataFrame({'Step': range(1, future_steps + 1), 'Predicted Close': predicted_values.flatten()})

# Print the future predictions table
print(future_df)

# Save the predictions table to a CSV file (optional)
future_df.to_csv('future_predictions.csv', index=False)
