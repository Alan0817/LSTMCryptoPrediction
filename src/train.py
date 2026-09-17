"""Training entry point and reusable functions for the unchanged LSTM classifier."""

from datetime import datetime, timezone

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from artifacts import save_model_artifact
from config import BATCH_SIZE, DATA_PATH, EPOCHS, LEARNING_RATE
from dataset import load_processed_data, prepare_datasets
from model import LSTMClassifier


def evaluate_classifier(model, X_test, y_test, batch_size=BATCH_SIZE, device=None):
    """Evaluate with the original 0.5 probability cutoff."""
    device = device or next(model.parameters()).device
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    predictions, actuals = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device)).squeeze()
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            actuals.extend(y_batch.numpy())
    return {
        "accuracy": accuracy_score(actuals, predictions),
        "classification_report": classification_report(actuals, predictions),
    }


def train_model(prepared, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, device=None):
    """Train the original two-layer LSTM using BCELoss and Adam defaults."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=prepared.X_train.shape[2]).to(device)
    train_loader = DataLoader(
        TensorDataset(prepared.X_train, prepared.y_train), batch_size=batch_size, shuffle=False
    )
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device)).squeeze()
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(train_loader)
        losses.append(average_loss)
        print("Epoch [{}/{}] Loss: {:.4f}".format(epoch + 1, epochs, average_loss))
    return model, losses


def run_training(data_path=DATA_PATH):
    """Prepare data, train, evaluate, and persist a reproducible model artifact."""
    prepared = prepare_datasets(load_processed_data(data_path))
    model, losses = train_model(prepared)
    evaluation = evaluate_classifier(model, prepared.X_test, prepared.y_test)
    training_metadata = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_sequences": len(prepared.X_train),
        "test_sequences": len(prepared.X_test),
        "final_loss": losses[-1],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accuracy": evaluation["accuracy"],
    }
    artifact_path = save_model_artifact(model, prepared.scaler, training_metadata)
    return model, evaluation, artifact_path


def main():
    _, evaluation, artifact_path = run_training()
    print("Model artifact saved to: {}".format(artifact_path))
    print("Accuracy: {:.4f}".format(evaluation["accuracy"]))
    print(evaluation["classification_report"])


if __name__ == "__main__":
    main()
