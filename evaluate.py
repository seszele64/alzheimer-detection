import torch
from model.alzheimers_model import AlzheimerNet
from utils.data_utils import get_data_loaders

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    total, correct = 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

def main():
    _, test_loader = get_data_loaders('/path/to/train', '/path/to/test')
    model = AlzheimerNet(num_classes=4)
    model.load_state_dict(torch.load('/path/to/model.pth'))
    
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
