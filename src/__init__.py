"""
Pacote src para o projeto de Classificação de Pneumonia em Raios-X com CNN.

Este pacote contém os módulos necessários para carregar dados, definir o modelo,
treinar, avaliar e visualizar os resultados da classificação de imagens de raio-X.
"""

# Você pode importar funções principais aqui para facilitar o acesso
from .data_loader import get_dataloaders, PneumoniaDataset
from .model import PneumoniaCNN
from .train import train_one_epoch, validate
from .evaluate import evaluate_model
from .visualize import plot_predictions, plot_training_history
