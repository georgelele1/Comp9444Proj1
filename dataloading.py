import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def train_and_validate(model, device, train_loader, val_loader, criterion, optimizer, epochs=10, save_path="my_model.pth"):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss={running_loss / len(train_loader):.4f} | Val Acc={accuracy:.2f}%")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

def test_model(model, device, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    data_dir = "Dataset"  # Should contain train, val, and test subfolders
    batch_size = 64
    num_workers = 4
    num_classes = 5
    epochs = 10
    save_path = "my_model.pth"

    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, num_workers)
    print("Loaded datasets!")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("Current device:", torch.cuda.current_device())
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_and_validate(model, device, train_loader, val_loader, criterion, optimizer, epochs, save_path)

    # Load best model and test
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_model(model, device, test_loader)

if __name__ == '__main__':
    main()
