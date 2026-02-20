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

## 📊 Estrutura do Projeto

A estrutura do repositório foi organizada para facilitar a navegação e reprodutibilidade:
