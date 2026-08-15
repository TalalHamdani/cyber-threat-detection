# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchmetrics import Accuracy

# Load preprocessed data
train_df = pd.read_csv('./data/labelled_train.csv')
test_df = pd.read_csv('./data/labelled_test.csv')
val_df = pd.read_csv('./data/labelled_validation.csv')

# View the first 5 rows of training set
train_df.head()

# Separate features and labels
X_train = train_df.drop(columns=['sus_label'])
y_train = train_df['sus_label']

X_test = test_df.drop(columns=['sus_label'])
y_test = test_df['sus_label']

X_val = val_df.drop(columns=['sus_label'])
y_val = val_df['sus_label']

# Scale features
std_scaler = StandardScaler()

X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)
X_val_scaled = std_scaler.transform(X_val)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype = torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype = torch.float32)

y_train_tensor = torch.tensor(y_train).reshape(-1,1)
y_test_tensor = torch.tensor(y_test).reshape(-1,1)
y_val_tensor = torch.tensor(y_val).reshape(-1,1)

# Define NN model
dataset = TensorDataset(X_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

input_dim = X_train_tensor.shape[1]
hidden_dim1 = 32
hidden_dim2 = 16
output_dim = 1

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim1),
    nn.ReLU(),
    nn.Linear(hidden_dim1, hidden_dim2),
    nn.ReLU(),
    nn.Linear(hidden_dim2, output_dim),
    nn.Sigmoid()
)

# Loss funtion and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 1e-4)

# Train and evaluate model
for epoch in range(10):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float().view(-1, 1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item(): .4f}")

# Initialize binary accuracy metric
accuracy = Accuracy(task="binary")
model.eval()

with torch.no_grad():
    train_preds = model(X_train_tensor)
    train_acc = accuracy(train_preds, y_train_tensor.int())

    val_preds = model(X_val_tensor)
    val_acc = accuracy(val_preds, y_val_tensor.int())

    test_preds = model(X_test_tensor)
    test_acc = accuracy(test_preds, y_test_tensor.int())

# Save validation accuracy
val_accuracy = int(round(val_acc.item() * 100))

# Print results
print("\n\nFinal Results:")
print(f"Training Accuracy:   {train_acc.item() * 100:.2f}%")
print(f"Validation Accuracy: {val_acc.item() * 100:.2f}% (Saved val_accuracy: {val_accuracy})")
print(f"Testing Accuracy:    {test_acc.item() * 100:.2f}%")