"""
Trabalho Prático de Redes Neurais e Aprendizado Profundo (ICMC/USP)

Problema: Classificação de Gêneros Cinematográficos usando Sinopse de Filmes
Modalidade: Texto (NLP)
Dataset: https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-multimodal-vision-and-nlp-genre-classification
Modelo: BERT + Classificador Linear (multi-label)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from transformers import AdamW
import torch.nn as nn

# Configurações globais
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
MODEL_NAME = "bert-base-uncased"

# Carregamento do dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['description', 'Genre'])
    df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
    return df


def main():
    print("Carregando dados...")
    df = load_data("data/IMDB_four_genre_larger_plot_description.csv")

if __name__ == "__main__":
    main()