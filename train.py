"""
This file contains the main training logic for the Alzheimer's detection model

It defines functions for training the model and the main execution flow.
"""

import torch
from model.alzheimers_model import AlzheimerNet
from utils.data_utils import get_data_loaders
from torch import optim, nn
import os


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    """
    Train the Alzheimer's detection model.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimization algorithm.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 25.

    Returns:
        model (nn.Module): The trained model.

    This function trains the model for the specified number of epochs,
    updating the model parameters based on the computed loss.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs} completed.')

    return model


def main():
    """
    Main function to set up and start the training process.

    This function initializes the data loaders, model, optimizer, and loss function,
    then starts the training process.
    """

    train_loader, _ = get_data_loaders('data/train', 'data/test')
    model = AlzheimerNet(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(
        model, train_loader, criterion, optimizer, num_epochs=25)

    # Save the trained model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(trained_model.state_dict(), 'saved_models/alzheimer_model.pth')
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
