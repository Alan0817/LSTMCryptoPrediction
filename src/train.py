import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from model import LSTMClassifier
from dataset import X_train, y_train, X_test, y_test



# DataLoader parameters
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(
    input_size=X_train.shape[2]
).to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Start training
EPOCHS = 100

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch).squeeze()

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# Evaluate on test set
model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch).squeeze()

        preds = (outputs > 0.5).float()

        predictions.extend(preds.cpu().numpy())
        actuals.extend(y_batch.numpy())

# Calculate accuracy and classification report
accuracy = accuracy_score(actuals, predictions)

print(f"Accuracy: {accuracy:.4f}")

print(classification_report(actuals, predictions))