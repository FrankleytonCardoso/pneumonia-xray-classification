import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.data_loader import get_dataloaders
from src.model import PneumoniaCNN
import os

def evaluate_model(model, test_loader, device, class_names=['NORMAL', 'PNEUMONIA']):
    """
    Avalia o modelo no conjunto de teste e retorna as predições e rótulos verdadeiros.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plota e salva a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Configurações
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    DATASET_PATH = "../data/chest_xray"
    MODEL_PATH = "../models/best_model.pth"
    REPORT_PATH = "../reports/figures/confusion_matrix.png"

    test_dir = os.path.join(DATASET_PATH, "test")

    # Carregar dados
    _, _, test_loader = get_dataloaders(
        train_dir="",  # Não precisamos do train/val aqui
        val_dir="",
        test_dir=test_dir,
        batch_size=BATCH_SIZE
    )

    # Carregar modelo treinado
    model = PneumoniaCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Modelo carregado com sucesso!")

    # Avaliar
    y_true, y_pred = evaluate_model(model, test_loader, DEVICE)

    # Calcular acurácia
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    # Relatório de classificação detalhado
    print("Relatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    # Plotar matriz de confusão
    plot_confusion_matrix(y_true, y_pred, class_names=['NORMAL', 'PNEUMONIA'], save_path=REPORT_PATH)
