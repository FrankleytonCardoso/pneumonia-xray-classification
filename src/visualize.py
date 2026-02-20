import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import PneumoniaDataset
from src.model import PneumoniaCNN
import os

def imshow(img, title=None):
    """
    Função auxiliar para exibir uma imagem tensor.
    """
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # Desnormalizar
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def plot_predictions(model, dataset, device, num_images=10, save_path=None):
    """
    Plota exemplos do dataset com as predições do modelo.
    - Verde: classificação correta
    - Vermelho: classificação incorreta
    """
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    indices = np.random.choice(len(dataset), num_images, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            # Adicionar dimensão de batch
            image_tensor = image.unsqueeze(0).to(device)

            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

            # Mover imagem para CPU para visualização
            image_cpu = image.cpu()

            # Definir cor do título
            color = 'green' if predicted.item() == label else 'red'
            title = f'V: {label} | P: {predicted.item()}'
            axes[i].set_title(title, color=color)

            # Exibir imagem
            imshow(image_cpu)
            axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualização de predições salva em: {save_path}")
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plota o histórico de treinamento (loss e acurácia por época).
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Treino')
    ax1.plot(epochs, val_losses, 'r-', label='Validação')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss durante o Treinamento')
    ax1.legend()
    ax1.grid(True)

    # Acurácia
    ax2.plot(epochs, train_accs, 'b-', label='Treino')
    ax2.plot(epochs, val_accs, 'r-', label='Validação')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia')
    ax2.set_title('Acurácia durante o Treinamento')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Histórico de treinamento salvo em: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Exemplo de uso (após o treinamento)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "../models/best_model.pth"
    TEST_DIR = "../data/chest_xray/test"
    SAVE_PATH = "../reports/figures/predictions.png"

    # Carregar modelo
    model = PneumoniaCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Carregar dataset de teste (sem transformações de aumento)
    _, test_transforms = get_transforms()  # Precisamos importar esta função
    from src.data_loader import get_transforms
    _, test_transforms = get_transforms()
    test_dataset = PneumoniaDataset(root_dir=TEST_DIR, transform=test_transforms)

    # Visualizar predições
    plot_predictions(model, test_dataset, DEVICE, num_images=10, save_path=SAVE_PATH)
