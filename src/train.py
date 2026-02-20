import torch
import torch.nn as nn
import torch.optim as optim
from src.model import PneumoniaCNN
from src.data_loader import get_dataloaders
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    # Configurações
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_PATH = "../data/chest_xray"  # Ajuste o caminho

    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")
    test_dir = os.path.join(DATASET_PATH, "test")

    # Carregar dados
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir, val_dir, test_dir, batch_size=BATCH_SIZE
    )

    # Inicializar modelo, loss e otimizador
    model = PneumoniaCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        # Salvar o melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../models/best_model.pth")
            print(f"Melhor modelo salvo com acurácia de validação: {best_val_acc:.4f}\n")

    print("Treinamento concluído!")
