import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18, densenet121, ResNet18_Weights, DenseNet121_Weights
import random
import pandas as pd

# Hyperparams
BATCH_SIZE = 64 # Use 64
IMG_SIZE = 128 # Use 128
EPOCHS = 50 # Use 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Using GPU
POKEMON_TYPES = [t.lower() for t in ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']]
NUM_CLASSES = len(POKEMON_TYPES)
TYPE_TO_IDX = {type: idx for idx, type in enumerate(POKEMON_TYPES)}
IDX_TO_TYPE = {idx: type for type, idx in TYPE_TO_IDX.items()}

pokemon_df = pd.read_csv("pokemonGen1.csv")
pokemon_name_to_type = pd.Series(pokemon_df['Type1'].str.lower().values, index=pokemon_df['Name'].str.lower()).to_dict()

directory = "./PokemonData"
image_paths = []
image_pokemon_names = []
image_labels = []  


# Go though every image file in each subfolder
for pokemon_name in os.listdir(directory):
    pokemon_folder = os.path.join(directory, pokemon_name)
    pokemon_name_lower = pokemon_name.lower()
    if os.path.isdir(pokemon_folder) and pokemon_name_lower in pokemon_name_to_type:
        type1 = pokemon_name_to_type[pokemon_name_lower]
        numerical_label = TYPE_TO_IDX[type1]
        for file in os.listdir(pokemon_folder):
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths.append(os.path.join(pokemon_folder, file))
                image_pokemon_names.append(pokemon_name)
                image_labels.append(numerical_label)
    elif os.path.isdir(pokemon_folder):
        print(f"Warning: Pokémon '{pokemon_name}' folder found but (case-insensitively) not in pokemonGen1.csv. Skipping.")

X = np.array(image_paths)
y = np.array(image_labels)


train_X, test_X, train_y, test_y, train_pokemon_names, test_pokemon_names = train_test_split(
    X, y, np.array(image_pokemon_names), test_size=0.2, random_state=42, stratify=y
)

# Borrowed Code
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Borrowed Function,Taken from Kraggle
class PokemonTypeDataset(Dataset):
    def __init__(self, image_paths, labels, pokemon_names, transform=None):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pokemon_names = pokemon_names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        pokemon_name = self.pokemon_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], pokemon_name

train_dataset = PokemonTypeDataset(train_X, train_y, train_pokemon_names, transform=train_transform)
test_dataset = PokemonTypeDataset(test_X, test_y, test_pokemon_names, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class PretrainedDenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

models = {
    "SimpleCNN": SimpleCNN(NUM_CLASSES),
    "ResNet18": PretrainedResNet18(NUM_CLASSES),
    "DenseNet121": PretrainedDenseNet(NUM_CLASSES)
}

results = {}
best_model = None
best_acc = 0

def train_and_evaluate(model, name):
    global best_model, best_acc
    model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_accuracies = []  # To store training accuracy per epoch

    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        correct_train = 0
        total_train = 0
        total_loss = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        epoch_end = time.time()
        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Train Acc: {train_accuracy * 100:.2f}% - Time: {epoch_end - epoch_start:.2f}s")

    total_time = time.time() - start_time
    print(f"\nFinished training {name} in {total_time:.2f} seconds.\n")

    model.eval()
    preds, targets, pokemon_names = [], [], []
    with torch.no_grad():
        for images, labels, names in test_loader:
            outputs = model(images.to(DEVICE))
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            pokemon_names.extend(list(names))

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    cm = confusion_matrix(targets, preds)
    mse = mean_squared_error(targets, preds)
    print(f"Test Accuracy ({name}): {acc * 100:.2f}%")
    print(f"F1 Score ({name}): {f1 * 100:.2f}%")
    print(f"Confusion Matrix ({name}):\n{cm}")
    print(f"Mean Squared Error ({name}): {mse:.4f}")

    torch.save(model.state_dict(), f"{name}.pth")

    if acc > best_acc:
        best_model = model
        best_acc = acc
        torch.save(model.state_dict(), f"best_model.pth")

    results[name] = {"accuracy": acc, "f1": f1, "confusion_matrix": cm, "mse": mse, "train_accuracies": train_accuracies}

    print(f"\n--- Sample Type 1 Guesses ({name}) ---")
    sample_indices = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))
    model.eval()
    for idx in sample_indices:
        image, true_label_idx, pokemon_name = test_dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(DEVICE))
            predicted_label_idx = torch.argmax(output, dim=1).item()
            predicted_type = IDX_TO_TYPE[predicted_label_idx]
            actual_type1 = pokemon_name_to_type.get(pokemon_name.lower(), "Not Found in CSV")
            true_type_from_dir = IDX_TO_TYPE[true_label_idx.item()]
            print(f"Pokemon: {pokemon_name}, Predicted Type: {predicted_type}, Actual Type1 (CSV): {actual_type1}")


# --- Run Models ---
for name, model in models.items():
    train_and_evaluate(model, name)

plt.figure(figsize=(18, 5))

# Test Accuracy Plot
plt.subplot(1, 3, 1)
accuracies = [res["accuracy"] * 100 for res in results.values()]
plt.bar(results.keys(), accuracies, color='skyblue')
plt.ylabel('Test Accuracy (%)')
plt.title('Pokémon Type1 Classification - Model Comparison (Test Accuracy)')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()

# F1 Score Plot
plt.subplot(1, 3, 2)
f1_scores = [res["f1"] * 100 for res in results.values()]
plt.bar(results.keys(), f1_scores, color='lightcoral')
plt.ylabel('F1 Score (%)')
plt.title('Pokémon Type1 Classification - Model Comparison (F1 Score)')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()

# Training Accuracy Plot
plt.subplot(1, 3, 3)
for name, res in results.items():
    epochs = range(1, len(res["train_accuracies"]) + 1)
    plt.plot(epochs, [acc * 100 for acc in res["train_accuracies"]], marker='o', label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy (%)')
plt.title('Training Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

print("\n--- Summary of Results ---")
for name, res in results.items():
    print(f"\nModel: {name}")
    print(f"  Accuracy: {res['accuracy'] * 100:.2f}%")
    print(f"  F1 Score: {res['f1'] * 100:.2f}%")
    print(f"  Confusion Matrix:\n{res['confusion_matrix']}")
    print(f"  Mean Squared Error: {res['mse']:.4f}")

print("\nBest Model:", "best_model.pth" if best_model else "None")