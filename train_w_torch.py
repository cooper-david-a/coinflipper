import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CoinCNN(nn.Module):
    """Simple CNN for heads/tails classification"""
    def __init__(self):
        super(CoinCNN, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv blocks
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Dense layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


from torch.utils.data import Dataset
from PIL import Image

class CoinDataset(Dataset):
    """Custom dataset that loads images from flat directory and labels from filenames"""
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Extract labels from filenames (heads_*.jpg -> 0, tails_*.jpg -> 1)
        self.labels = []
        for filename in self.image_files:
            if filename.lower().startswith('heads'):
                self.labels.append(0)
            elif filename.lower().startswith('tails'):
                self.labels.append(1)
            else:
                raise ValueError(f"Cannot determine label for {filename}. "
                               "Filename must start with 'heads' or 'tails'")
        
        # Class mapping
        self.class_to_idx = {'heads': 0, 'tails': 1}
        self.idx_to_class = {0: 'heads', 1: 'tails'}
        
        print(f"Loaded {len(self.image_files)} images")
        heads_count = sum(1 for l in self.labels if l == 0)
        tails_count = sum(1 for l in self.labels if l == 1)
        print(f"  Heads: {heads_count}")
        print(f"  Tails: {tails_count}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def train_model(data_dir, output_model_path='coin_classifier.pth',
                output_onnx_path='coin_classifier.onnx', epochs=100, batch_size=32,
                val_split=0.2):
    """
    Train CNN on coin images
    
    Directory structure expected:
    data_dir/
        heads_01.jpg
        heads_02.jpg
        heads_01_rot00.jpg
        heads_01_rot01.jpg
        ...
        tails_01.jpg
        tails_02.jpg
        tails_01_rot00.jpg
        tails_01_rot01.jpg
        ...
    
    Labels are determined from filenames (heads_* or tails_*).
    Images will be randomly split into train/val sets.
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load full dataset with train transforms initially
    full_dataset = CoinDataset(data_dir, transform=train_transform)
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Random split
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create separate datasets with appropriate transforms
    train_dataset = CoinDataset(data_dir, transform=train_transform)
    val_dataset = CoinDataset(data_dir, transform=val_transform)
    
    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices.indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices.indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    print(f"\nTotal images: {dataset_size}")
    print(f"Training images: {train_size}")
    print(f"Validation images: {val_size}")
    print(f"Classes: {full_dataset.class_to_idx}")
    
    # Create model
    model = CoinCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as {output_model_path}")
    
    # Load best model for export
    model.load_state_dict(torch.load(output_model_path))
    model.eval()
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    dummy_input = torch.randn(1, 1, 100, 100).to(device)
    
    # Use dynamo=False for compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,  # Use newer opset version
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False  # Use legacy exporter to avoid compatibility issues
    )
    print(f"Model exported to {output_onnx_path}")
    
    return model


if __name__ == "__main__":
    # Set your data directory here (flat directory with images)
    data_dir = './templates/processed/rotated'  # UPDATE THIS - should contain heads_*.jpg and tails_*.jpg files
    
    # Output paths
    output_model_path = 'coin_classifier.pth'
    output_onnx_path = 'coin_classifier.onnx'
    
    # Train model with 80/20 train/val split
    model = train_model(
        data_dir=data_dir,
        output_model_path=output_model_path,
        output_onnx_path=output_onnx_path,
        epochs=100,
        batch_size=32,
        val_split=0.2  # 20% for validation
    )
    
    print("\nTraining complete!")