import torch
from model.alzheimers_model import AlzheimerNet
from utils.data_utils import get_data_loaders
from torch import optim, nn

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
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

def main():
    train_loader, _ = get_data_loaders('/path/to/train', '/path/to/test')
    model = AlzheimerNet(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, optimizer, num_epochs=25)

if __name__ == "__main__":
    main()
