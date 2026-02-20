# Pneumonia-Xray-Classification
Classificação de Pneumonia em imagens de Raio-X utilizando Redes Neurais Convolucionais (CNN) com PyTorch e Transfer Learning (ResNet18). Projeto de estudo para diagnóstico médico automatizado.

# Classificação de Pneumonia em Raios-X com Redes Neurais Convolucionais (CNN)

[![Medium](https://img.shields.io/badge/Medium-Artigo-blue)](link_para_o_seu_artigo)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-orange)](link_para_o_seu_notebook_original_no_kaggle)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)

## 📋 Sobre o Projeto

A pneumonia é uma das principais causas de mortalidade global, especialmente em regiões com acesso limitado a serviços de saúde. O diagnóstico precoce é essencial para um tratamento eficaz, mas a interpretação manual de radiografias pode ser subjetiva e demorada.

Este projeto propõe uma solução automatizada baseada em **Redes Neurais Convolucionais (CNNs)** para classificar imagens de raio-X de tórax, distinguindo entre pacientes saudáveis (`NORMAL`) e aqueles com pneumonia (`PNEUMONIA`). O objetivo é explorar como a inteligência artificial pode contribuir para diagnósticos mais rápidos e precisos, auxiliando profissionais de saúde.

O desenvolvimento completo e os resultados detalhados estão descritos neste [artigo no Medium](https://medium.com/@kleyto.cardoso/pneumonia-classification-on-x-rays-with-convolutional-neural-networks-cnn-0214061c8b80).

## 🎯 Objetivos

*   Automatizar a classificação de imagens de raio-X para detecção de pneumonia.
*   Demonstrar a aplicação prática de *transfer learning* com a arquitetura ResNet18.
*   Fornecer um modelo de código aberto e acessível para a comunidade.

## 🛠️ Tecnologias e Ferramentas Utilizadas

*   **Linguagem:** Python 3.10
*   **Principais Bibliotecas:**
    *   **PyTorch:** Framework principal para construção e treinamento da CNN.
    *   **Torchvision:** Para modelos pré-treinados (ResNet18) e transformações de imagem.
    *   **OpenCV (cv2):** Para processamento de imagem (leitura, redimensionamento).
    *   **Matplotlib:** Para visualização de dados e resultados.
    *   **scikit-learn:** Para cálculo de métricas de avaliação (acurácia, matriz de confusão).
*   **Ambiente de Treinamento:** [Kaggle](https://www.kaggle.com/) com GPU NVIDIA.
*   **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) do Kaggle.

## 📁 Estrutura do Projeto

```
projeto/
├── data/                    # Dados do projeto
│   └── README.md
├── notebooks/               # Notebooks para análise exploratória
│   └── notebook-ds-project.ipynb
├── src/                     # Código fonte principal
│   ├── __init__.py
│   ├── data_loader.py      # Carregamento de dados
│   ├── evaluate.py          # Avaliação de modelos
│   ├── model.py             # Definição dos modelos
│   ├── train.py             # Treinamento
│   └── visualize.py         # Visualizações
├── .gitignore
├── ACKNOWLEDGMENTS.md
├── README.md                # README na raiz
├── notebook-ds-project.ipynb
└── requirements.txt
```

## 🚀 Como Executar o Projeto

### Pré-requisitos

*   Python 3.8 ou superior.
*   `pip` (gerenciador de pacotes do Python).
*   (Opcional) GPU com CUDA para treinamento mais rápido.

### Passo a Passo

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/FrankleytonCardoso/pneumonia-xray-classification
    cd chest-xray-pneumonia-cnn
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Baixe o dataset:**
    *   Acesse o [dataset no Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
    *   Coloque os arquivos na pasta `data/`. A estrutura esperada é:
        ```
        data/
            chest_xray/
                train/
                    NORMAL/
                    PNEUMONIA/
                val/
                    NORMAL/
                    PNEUMONIA/
                test/
                    NORMAL/
                    PNEUMONIA/
        ```

4.  **Execute o treinamento:**
    ```bash
    python src/train.py
    ```
    (Os logs e o modelo treinado serão salvos em `reports/logs/` e `models/`, respectivamente.)

5.  **Avalie o modelo treinado:**
    ```bash
    python src/evaluate.py
    ```

## 📈 Principais Resultados

*   **Modelo:** ResNet18 pré-treinado com *transfer learning*.
*   **Acurácia no Teste:** 76,12%.
*   **Técnicas de Pré-processamento:** CLAHE para melhoria de contraste, normalização e *data augmentation* (rotação, inversão horizontal) para combater overfitting.

Para uma análise mais aprofundada, leia o [artigo completo no Medium](link_para_o_seu_artigo).

## 🤝 Como Contribuir

Contribuições são sempre bem-vindas! Sinta-se à vontade para abrir uma *issue* ou um *pull request*.

1.  Faça um *fork* do projeto.
2.  Crie uma *branch* para sua feature (`git checkout -b feature/nova-feature`).
3.  Faça o *commit* das suas alterações (`git commit -m 'Adiciona nova feature'`).
4.  Faça o *push* para a *branch* (`git push origin feature/nova-feature`).
5.  Abra um *Pull Request*.

## ✉️ Contato

*   **Autor:** Frankleyton Cardoso de Oliveira
*   **Medium:** [@kleyto.cardoso](https://medium.com/@kleyto.cardoso)
*   **LinkedIn:** https://www.linkedin.com/in/frankleyton-oliveira-22b72a112/
