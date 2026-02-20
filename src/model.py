import torch
import torch.nn as nn
import torchvision.models as models

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        # Carregar modelo pré-treinado
        self.model = models.resnet18(pretrained=True)

        # Congelar os parâmetros das camadas iniciais (opcional)
        for param in self.model.parameters():
            param.requires_grad = False

        # Substituir a última camada fully connected para o número de classes desejado
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
