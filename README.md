# DL- Developing a Recurrent Neural Network Model for Stock Prediction
# Name:Yuvaram S

# Register Number: 212224230315
## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

The objective of this experiment is to develop a Recurrent Neural Network (RNN) model using PyTorch to predict stock prices from historical closing-price data. The provided trainset.csv and testset.csv datasets are used for training and testing the model. The closing prices are normalized using MinMaxScaler, and sequences of 60 previous stock prices are created to predict the next stock price. The RNN model is trained using MSE Loss and the Adam optimizer, and its performance is evaluated by comparing the actual and predicted stock prices.

## DESIGN STEPS
### STEP 1: 

Load the trainset.csv and testset.csv datasets and extract the Close column as the stock-price data.

### STEP 2: 

Normalize the stock prices using MinMaxScaler, fitting the scaler only on the training data to avoid data leakage.


### STEP 3: 

Create time-series sequences using 60 previous closing prices as input and the next closing price as the target value.

### STEP 4: 

Convert the prepared sequences into PyTorch tensors and create a TensorDataset and DataLoader for efficient model training.

### STEP 5: 

Define an RNN model with an input layer, two RNN layers with 64 hidden units, and a fully connected output layer. Train the model using MSELoss and the Adam optimizer.

### STEP 6: 

Test the trained RNN model using testset.csv, convert the predictions back to the original price scale, and plot the training loss and actual versus predicted stock prices.

## PROGRAM

### Name:Yuvaram S

### Register Number: 212224230315

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Step 1: Load Dataset

df_train = pd.read_csv("trainset.csv")
df_test = pd.read_csv("testset.csv")

print("Training Dataset:")
print(df_train.head())

print("\nTesting Dataset:")
print(df_test.head())

print("\nTraining Data Shape:", df_train.shape)
print("Testing Data Shape:", df_test.shape)


# Step 2: Preprocess the Data

train_prices = df_train["Close"].values.reshape(-1, 1)
test_prices = df_test["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)


# Step 3: Create Sequences

def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)


seq_length = 60

x_train, y_train = create_sequences(
    scaled_train,
    seq_length
)

x_test, y_test = create_sequences(
    scaled_test,
    seq_length
)

print("\nSequence Information:")
print("X Train Shape:", x_train.shape)
print("Y Train Shape:", y_train.shape)
print("X Test Shape:", x_test.shape)
print("Y Test Shape:", y_test.shape)


# Step 4: Convert Data into PyTorch Tensors

x_train_tensor = torch.tensor(
    x_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
)

x_test_tensor = torch.tensor(
    x_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
)

train_dataset = TensorDataset(
    x_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Step 5: Define RNN Model

class RNNModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        out, _ = self.rnn(x)

        out = self.fc(
            out[:, -1, :]
        )

        return out


model = RNNModel()


# Step 6: Select Device

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

print("\nUsing Device:", device)

print("\nRNN Model:")
print(model)


# Loss Function and Optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Step 7: Train the Model

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=20
):

    train_losses = []

    print("\nName: Yuvaram S")
    print("Register Number: 212224230315")

    print("\nTraining the RNN Model...\n")

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        train_losses.append(
            average_loss
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {average_loss:.6f}"
        )

    return train_losses


train_losses = train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=20
)


# Step 8: Training Loss Plot

plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "Training Loss Over Epochs"
)

plt.legend()
plt.grid()

plt.show()


# Step 9: Make Predictions on Test Data

model.eval()

with torch.no_grad():

    predicted = model(
        x_test_tensor.to(device)
    ).cpu().numpy()

actual = y_test_tensor.numpy()

predicted_prices = scaler.inverse_transform(
    predicted
)

actual_prices = scaler.inverse_transform(
    actual
)


# Step 10: Plot Actual vs Predicted Prices

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Price"
)

plt.xlabel("Time")
plt.ylabel("Stock Price")

plt.title(
    "Stock Price Prediction using RNN"
)

plt.legend()
plt.grid()

plt.show()


# Step 11: Display Predictions

print("\nName: DHINESHKUMAR L")
print("Register Number: 212224230066")

print("\nPredictions on Test Data:\n")

for i in range(
    min(10, len(predicted_prices))
):

    print(
        f"Actual Price: "
        f"{actual_prices[i][0]:.2f} | "
        f"Predicted Price: "
        f"{predicted_prices[i][0]:.2f}"
    )

print("\nFinal Prediction:")

print(
    f"Predicted Price: "
    f"{predicted_prices[-1][0]:.2f}"
)

print(
    f"Actual Price: "
    f"{actual_prices[-1][0]:.2f}"
)


# Result

print(
    "\nRESULT: The RNN model for stock price "
    "prediction was successfully implemented "
    "and tested using historical closing-price data."
)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="620" height="522" alt="Screenshot 2026-08-27 105153" src="https://github.com/user-attachments/assets/d2783060-9fdc-4f79-a5bc-6719b1645a60" />

<img width="1070" height="578" alt="Screenshot 2026-08-25 123937" src="https://github.com/user-attachments/assets/48febe67-92ae-434d-926f-1a8cbd27fab1" />



## True Stock Price, Predicted Stock Price vs time

<img width="1272" height="697" alt="Screenshot 2026-08-25 124129" src="https://github.com/user-attachments/assets/6e01e28f-ba4b-4e5a-af93-4aed5129e2e7" />



### Predictions

<img width="1609" height="103" alt="image" src="https://github.com/user-attachments/assets/d1a8530a-46de-4bdf-98cb-7b0c38db3e3d" />



## RESULT

Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been implemented successfully.
