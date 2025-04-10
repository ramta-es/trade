import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset, batch_size=16, lr=1e-4, num_epochs=10, device='cuda'):
        """
        Initialize the Trainer object.

        Args:
            model (nn.Module): The model to train (e.g., SwinTransformerSys or SwinTransformerVideoPredictor).
            dataset (torch.utils.data.Dataset): The dataset for training and evaluation.
            batch_size (int): Batch size for training and evaluation.
            lr (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs to train.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

        # Split dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()  # Adjust based on your task
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        """
        Train the model.
        """
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            for inputs, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(self.train_loader):.4f}, Accuracy = {100.*correct/total:.2f}%")

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Track metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f"Validation: Loss = {val_loss/len(self.val_loader):.4f}, Accuracy = {100.*correct/total:.2f}%")