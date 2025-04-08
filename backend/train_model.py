import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(num_classes):
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    
    return model.to(DEVICE)

def train_model(train_dir, validation_dir):
    # Data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'validation': datasets.ImageFolder(validation_dir, data_transforms['validation'])
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'validation': DataLoader(image_datasets['validation'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    # Create model
    num_classes = len(image_datasets['train'].classes)
    model = create_model(num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
    
    return model, image_datasets['train'].class_to_idx

def save_model_and_classes(model, class_to_idx, output_dir='models'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }, os.path.join(output_dir, 'plant_disease_model.pth'))
    
    # Save class indices
    with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f)

if __name__ == '__main__':
    # Update these paths to point to your dataset directories
    train_dir = '../data/organized/train'
    validation_dir = '../data/organized/validation'
    
    print(f"Using device: {DEVICE}")
    print("Starting model training...")
    model, class_to_idx = train_model(train_dir, validation_dir)
    
    print("Saving model and class indices...")
    save_model_and_classes(model, class_to_idx)
    
    print("Training complete!") 