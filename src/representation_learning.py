import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Creating custom BasicBlock to match the pre-trained model
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        
        if stride != 1 or inplanes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)

        return out

# Define a ResNet architecture that matches the pre-trained model
class ResNet(torch.nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # First layer adapted for CIFAR-10
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        
        # First block may have a stride
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

# Define the SimCLR model with ResNet backbone and projection head
class SimCLR(torch.nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        
        # Create a custom ResNet-18 backbone with matching architecture
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        
        # Projection head
        self.contrastive_head = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, feature_dim)
        )
    
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        if return_features:
            return features
        out = self.contrastive_head(features)
        return out

def load_pretrained_model(model_path):
    # Initialize the model
    model = SimCLR()
    
    # Load the pre-trained weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Load the state dict directly
    model.load_state_dict(state_dict)
    
    print("Pre-trained model loaded successfully.")
    return model

def get_data_loaders():
    # Use established CIFAR-10 data statistics for normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            # Get features from the backbone (before the projection head)
            outputs = model(images, return_features=True)
            
            # Apply L2 normalization to each feature vector
            normalized_features = torch.nn.functional.normalize(outputs, p=2, dim=1)
            
            features.append(normalized_features.cpu().numpy())
            labels.append(targets.numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

def main():
    # Load the pre-trained model
    model_path = './models/simclr_cifar-10.pth.tar'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model file not found: {model_path}")
    
    model = load_pretrained_model(model_path)
    model = model.to(device)
    
    # Get CIFAR-10 data loaders
    train_loader, test_loader = get_data_loaders()
    
    # Extract features from the training and test sets
    print("Extracting features from the training set...")
    train_features, train_labels = extract_features(model, train_loader)
    
    print("Extracting features from the test set...")
    test_features, test_labels = extract_features(model, test_loader)
    
    # Save the extracted features and labels
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # Create a directory for saving the features
    os.makedirs('../features', exist_ok=True)
    
    # Save the features and labels
    with open('./features/normalized_train_features.pkl', 'wb') as f:
        pickle.dump({'features': train_features, 'labels': train_labels}, f)
    
    with open('./features/normalized_test_features.pkl', 'wb') as f:
        pickle.dump({'features': test_features, 'labels': test_labels}, f)
    
    print("Features extracted and saved successfully.")

if __name__ == "__main__":
    main() 