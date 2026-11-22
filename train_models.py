"""
Multimodal Food Quality & Safety AI - PyTorch GPU Training Pipeline

Branch A: Electrochemical MIP Sensor ML Model
Branch B: PyTorch Transfer Learning & Fine-Tuning on GTX 1650 GPU
"""

import os
import numpy as np
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import joblib
from pathlib import Path

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ============================================================================
# BRANCH A: ELECTROCHEMICAL MIP SENSOR ML MODEL
# ============================================================================

print("\n" + "=" * 80)
print("BRANCH A: Electrochemical MIP Sensor ML Model")
print("=" * 80)

def generate_mip_sensor_data(n_samples=1500):
    """Generate synthetic electrochemical sensor data based on domain knowledge"""
    np.random.seed(42)
    
    peak_current_uA = np.zeros(n_samples)
    peak_voltage_V = np.zeros(n_samples)
    impedance_ohm = np.zeros(n_samples)
    ph_level = np.zeros(n_samples)
    labels = np.zeros(n_samples, dtype=int)
    
    # Class 0: Safe/Clean (600 samples)
    n_class0 = 600
    peak_current_uA[:n_class0] = np.random.normal(25, 5, n_class0)
    peak_voltage_V[:n_class0] = np.random.normal(0.45, 0.05, n_class0)
    impedance_ohm[:n_class0] = np.random.normal(5000, 800, n_class0)
    ph_level[:n_class0] = np.random.normal(6.5, 0.8, n_class0)
    labels[:n_class0] = 0
    
    # Class 1: Pesticide Residue (450 samples)
    n_class1 = 450
    base = n_class0
    peak_current_uA[base:base+n_class1] = np.random.normal(18, 4, n_class1)
    peak_voltage_V[base:base+n_class1] = np.random.normal(0.55, 0.08, n_class1)
    impedance_ohm[base:base+n_class1] = np.random.normal(3500, 600, n_class1)
    ph_level[base:base+n_class1] = np.random.normal(5.8, 0.7, n_class1)
    labels[base:base+n_class1] = 1
    
    # Class 2: Spoiled/Mycotoxin (450 samples)
    n_class2 = 450
    base = n_class0 + n_class1
    peak_current_uA[base:base+n_class2] = np.random.normal(12, 3, n_class2)
    peak_voltage_V[base:base+n_class2] = np.random.normal(0.65, 0.1, n_class2)
    impedance_ohm[base:base+n_class2] = np.random.normal(2000, 400, n_class2)
    ph_level[base:base+n_class2] = np.random.normal(4.5, 0.9, n_class2)
    labels[base:base+n_class2] = 2
    
    # Add noise
    peak_current_uA += np.random.normal(0, 1, n_samples)
    peak_voltage_V += np.random.normal(0, 0.02, n_samples)
    impedance_ohm += np.random.normal(0, 100, n_samples)
    ph_level += np.random.normal(0, 0.2, n_samples)
    
    df = pd.DataFrame({
        'peak_current_uA': np.clip(peak_current_uA, 1, 50),
        'peak_voltage_V': np.clip(peak_voltage_V, 0.2, 1.0),
        'impedance_ohm': np.clip(impedance_ohm, 500, 8000),
        'pH_level': np.clip(ph_level, 2.0, 9.0),
        'safety_class': labels
    })
    
    return df

# Generate and save sensor data
print("\nGenerating synthetic MIP sensor dataset...")
mip_data = generate_mip_sensor_data(1500)
os.makedirs('data', exist_ok=True)
mip_data.to_csv('data/mip_sensor_data.csv', index=False)
print(f"Saved {len(mip_data)} sensor records to data/mip_sensor_data.csv")

# Prepare features and labels
X = mip_data[['peak_current_uA', 'peak_voltage_V', 'impedance_ohm', 'pH_level']].values
y = mip_data['safety_class'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/sensor_rf_model.pkl')
joblib.dump(scaler, 'models/sensor_scaler.pkl')
print("Saved models/sensor_rf_model.pkl")
print("Saved models/sensor_scaler.pkl")

# ============================================================================
# BRANCH B: PYTORCH TRANSFER LEARNING & FINE-TUNING
# ============================================================================

print("\n" + "=" * 80)
print("BRANCH B: PyTorch Transfer Learning & Fine-Tuning")
print("=" * 80)

# Define transforms
IMG_SIZE = 224
BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Collect all class folders
print("\nIndexing image directories...")
fruits_dir = Path('Fruits')
vegetables_dir = Path('Vegetables')

def collect_image_files(folder_path, max_images=250):
    """Collect image files from a folder, capped at max_images"""
    extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(extensions):
                images.append(os.path.join(root, file))
                if len(images) >= max_images:
                    return images[:max_images]
    return images

# Collect classes with clean names
class_folders = []
class_labels = []

# Process Fruits directory
if fruits_dir.exists():
    for class_name in sorted(os.listdir(fruits_dir)):
        class_path = fruits_dir / class_name
        if class_path.is_dir():
            class_folders.append(str(class_path))
            # Clean name: convert "FreshApple" -> "fresh_apple", "RottenBanana" -> "rotten_banana"
            clean_name = class_name.lower().replace('fresh', 'fresh').replace('rotten', 'rotten')
            class_labels.append(clean_name)

# Process Vegetables directory
if vegetables_dir.exists():
    for class_name in sorted(os.listdir(vegetables_dir)):
        class_path = vegetables_dir / class_name
        if class_path.is_dir():
            class_folders.append(str(class_path))
            clean_name = class_name.lower().replace('fresh', 'fresh').replace('rotten', 'rotten')
            class_labels.append(clean_name)

print(f"Found {len(class_folders)} classes:")
for i, (folder, label) in enumerate(zip(class_folders, class_labels)):
    img_count = len(collect_image_files(folder, max_images=10))
    print(f"  {i+1}. {label}: ~{img_count} images")

# Create custom dataset
class FoodImageDataset(Dataset):
    def __init__(self, folders, labels, transform=None, max_per_class=250):
        self.samples = []
        self.transform = transform
        
        for folder, label in zip(folders, labels):
            images = collect_image_files(folder, max_images=max_per_class)
            for img_path in images:
                self.samples.append((img_path, label))
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Total samples: {len(self.samples)}")
        print(f"Number of classes: {len(self.label_to_idx)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        return image, label_idx

# Split into train/val
all_folders = class_folders
all_labels = class_labels

# Create dataset
dataset = FoodImageDataset(all_folders, all_labels, transform=train_transforms)

# Split indices
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=[dataset.samples[i][1] for i in indices])

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Build model
print("\nBuilding MobileNetV3 Large model...")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.Hardswish(),
    nn.Dropout(0.3),
    nn.Linear(256, len(dataset.label_to_idx)),
    nn.LogSoftmax(dim=1)
)

# Move to GPU
model = model.to(device)
print(f"Model moved to {device}")

# ============================================================================
# TRAINING
# ============================================================================

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / total

# Train for 4 epochs (frozen)
print("\nTraining classification head for 4 epochs (frozen)...")
for epoch in range(4):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/4 - Train: {train_loss:.4f}/{train_acc:.4f}, Val: {val_loss:.4f}/{val_acc:.4f}")

# ============================================================================
# FINE-TUNING
# ============================================================================

print("\nFine-tuning last 3 inverted residual blocks...")

# Unfreeze last 3 blocks (about 20-30 layers)
blocks_to_unfreeze = 3
feature_blocks = list(model.features.children())
for block in feature_blocks[-blocks_to_unfreeze:]:
    for param in block.parameters():
        param.requires_grad = True

# Recreate optimizer with lower LR
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train for 4 epochs (fine-tuning)
for epoch in range(4):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch+1}/4 - Train: {train_loss:.4f}/{train_acc:.4f}, Val: {val_loss:.4f}/{val_acc:.4f}")

# Final validation accuracy
final_val_acc = val_acc * 100
print(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\nSaving fine-tuned model...")
os.makedirs('models', exist_ok=True)

# Save model with class information
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': dataset.label_to_idx,
    'idx_to_class': dataset.idx_to_label,
    'input_size': IMG_SIZE,
    'device': str(device)
}, 'models/food_cnn_finetuned.pth')

print("Saved models/food_cnn_finetuned.pth")
print(f"Class mapping saved: {len(dataset.label_to_idx)} classes")

# Verify model loads
print("\nVerifying model save/load...")
checkpoint = torch.load('models/food_cnn_finetuned.pth', map_location=device)
model_loaded = models.mobilenet_v3_large(weights=None)
num_features = model_loaded.classifier[0].in_features
model_loaded.classifier = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.Hardswish(),
    nn.Dropout(0.3),
    nn.Linear(256, len(checkpoint['class_to_idx'])),
    nn.LogSoftmax(dim=1)
)
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded.to(device)
model_loaded.eval()

print("✅ Model verification successful!")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nModel Files Created:")
print("  ✓ models/sensor_rf_model.pkl (Random Forest classifier)")
print("  ✓ models/sensor_scaler.pkl (StandardScaler)")
print("  ✓ models/food_cnn_finetuned.pth (PyTorch MobileNetV3)")
print("  ✓ data/mip_sensor_data.csv (Synthetic sensor dataset)")

print("\nValidation Results:")
print(f"  • Sensor Model Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Final CNN Validation Accuracy: {final_val_acc:.2f}%")

print("\n" + "=" * 80)
