import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      
        self.pool = nn.MaxPool2d(2, 2)     
        self.conv2 = nn.Conv2d(6, 16, 5)    
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001,
            device='cpu', val_every=1, patience=5, out_path="best_model.pth"):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float('inf')
        counter = 0  

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = 100.0 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_loss, val_acc = self.evaluate(val_loader, criterion, device)
                print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), out_path)
                    counter = 0 
                    print(f"New best model saved (val_loss = {val_loss:.4f})")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        break
            elif val_loader is None:
                torch.save(self.state_dict(), out_path)
                print(f"Model saved to {out_path} (no validation).")

    def evaluate(self, dataloader, criterion=None, device='cpu'):
        """Evaluate the model on a DataLoader (e.g., test set)."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.eval()
        self.to(device)
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(dataloader.dataset)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
