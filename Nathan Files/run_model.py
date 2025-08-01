import pandas as pd
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import model_setup
import time

class ImageClassificationDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_dir, row['file name'])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row['label_idx'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(dspath, batch_size, num_workers):
    df = pd.read_csv("metadata.csv")
    image_dir = "grouped_ds"
    random_state = 42

    df['label_idx'] = df['class'].astype('category').cat.codes

    test_size = 0.2
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label_idx'],
        random_state=random_state
    )

    val_size = 1 - model_setup.train_val_split
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['label_idx'],
        random_state=random_state
    )

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
    ])

    train_dataset = ImageClassificationDataset(train_df, image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_dataset = ImageClassificationDataset(val_df, image_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_dataset = ImageClassificationDataset(test_df, image_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader

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
    data_dir = "Dataset"
    batch_size = model_setup.batch_size
    num_workers = 6
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, num_workers)
    print("Loaded datasets!")

    model = model_setup.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Memory (GB): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
        print(f"Multiprocessors: {torch.cuda.get_device_properties(0).multi_processor_count}")

    criterion = model_setup.loss_func
    optimizer = model_setup.optimizer
    weights_init = model_setup.weights_init
    scheduler = model_setup.scheduler

    model.apply(weights_init)

    print("Start training...")

    model = model.to(device)

    for epoch in range(1, model_setup.epochs+1):
        train_start = time.time()
        model.train()
        start_time = time.time()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        total_transfer_time = 0.0
        total_forward_time = 0.0
        num_batches = 0

        for images, labels in loop:
            transfer_start = time.time()
            images, labels = images.to(device), labels.to(device)
            total_transfer_time  += time.time() - transfer_start

            forward_start = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            total_forward_time += time.time() - forward_start
            num_batches += 1

        train_time = time.time() - train_start


        valid_start = time.time()
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
        valid_time = time.time() - valid_start
        print(f"Epoch {epoch}: Loss={running_loss / len(train_loader):.4f} | Val Acc={accuracy:.2f}%")
        print(f"Avg transfer time: {total_transfer_time / num_batches}s | Avg forward/backward time: {total_forward_time / num_batches}s | train time: {train_time:.4f}s | valid time: {valid_time:.4f}s")

    save_path = "model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

    # Load best model and test
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_model(model, device, test_loader)

if __name__ == '__main__':
    main()
