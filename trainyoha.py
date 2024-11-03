import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

data_dir = "train_data"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Load ResNet-50 model with pretrained weights and modify the final layer
CNN = models.resnet50(weights="IMAGENET1K_V2")  
CNN.fc = nn.Linear(CNN.fc.in_features, 3)  

CNN = CNN.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR every 5 epochs

# Training parameters
num_epochs = 10
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

# Training and evaluation function
def train_and_evaluate(model, criterion, optimizer, scheduler, num_epochs=10):
    start_time = time.time()

    for epoch in range(num_epochs):

        # Training 
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_losses.append(train_loss / total)
        train_accuracies.append(correct / total)

        # Testing 
        model.eval()
        test_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                print(predicted)
        
        test_losses.append(test_loss / total)
        test_accuracies.append(correct / total)

        scheduler.step()  # Update learning rate

        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}, '
              f'Time: {elapsed_time:.2f}s')

# Train and evaluate
if __name__ == "__main__":
    train_and_evaluate(CNN, criterion, optimizer, scheduler, num_epochs)



torch.save(CNN.state_dict(), "model.pth")
