"""
IMDB Movie Genre Classifier (Texto) - Versão Final Comentada

📌 Descrição:
Este script implementa um classificador multilabel de gêneros de filmes com base em suas descrições textuais (sinopses).
O modelo utiliza a arquitetura DistilBERT, um modelo pré-treinado da HuggingFace Transformers, para representar semanticamente
as sinopses e um classificador linear para prever os gêneros associados.

📌 Arquitetura:
- Tokenização com DistilBERTTokenizerFast
- Modelo base: DistilBERT + camada linear
- Treinamento com BCEWithLogitsLoss com pesos balanceados
- Otimizador: AdamW
- Scheduler: linear com warmup
- Avaliação: F1-score com ajuste automático de limiares por classe

📌 Dados:
O conjunto de dados utilizado é um CSV com colunas:
- `description`: texto da sinopse do filme
- `genre`: gêneros do filme separados por vírgulas (ex: "comedy, action")

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Configurações globais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 8
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# Dataset customizado para PyTorch
class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(self.labels[idx])
        }

# Função para carregar e preparar os dados
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'description' not in df.columns or 'genre' not in df.columns:
        raise ValueError("Esperado colunas 'description' e 'genre'.")
    df = df.dropna(subset=['description', 'genre'])
    df['genre'] = df['genre'].apply(lambda x: [g.strip() for g in x.split(',')])
    return df

# Cria um DataLoader com PyTorch
def create_dataloader(texts, labels, tokenizer, max_len, batch_size, shuffle):
    dataset = MovieDataset(texts, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())

# Modelo baseado em DistilBERT com uma camada linear para classificação multilabel
class GenreClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(self.dropout(pooled))

# Função de treinamento para uma época com clipping e scheduler
def train_epoch(model, data_loader, loss_fn, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(data_loader, desc="Treinando", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

# Avaliação do modelo (sem atualização dos pesos)
def eval_model(model, data_loader):
    model.eval()
    predictions, real_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validando", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs)
            predictions.append(preds.cpu().numpy())
            real_labels.append(labels.cpu().numpy())
    return np.vstack(predictions), np.vstack(real_labels)

# Encontra o melhor limiar (threshold) de decisão por classe baseado no F1
def find_best_thresholds(y_true, y_probs, thresholds=np.arange(0.1, 0.91, 0.05)):
    best_thresholds = []
    for i in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, i], (y_probs[:, i] >= t).astype(int), zero_division=0) for t in thresholds]
        best_thresholds.append(thresholds[np.argmax(scores)])
    return np.array(best_thresholds)

# Gera relatório com métricas usando os thresholds ajustados
def get_metrics_with_thresholds(preds, labels, thresholds, mlb):
    preds_binary = (preds >= thresholds).astype(int)
    print("\nRelatório de Classificação:")
    print(classification_report(labels, preds_binary, target_names=mlb.classes_, zero_division=0))

# Função principal que executa todo o pipeline
def main():
    # Carrega e binariza os gêneros
    df = load_data("data/IMDB_four_genre_larger_plot_description.csv")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['genre'])

    # Tokenizador do modelo
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Split de treino e validação
    X_train, X_val, y_train, y_val = train_test_split(df['description'], labels, test_size=0.2, random_state=42)

    # Loaders
    train_loader = create_dataloader(X_train.tolist(), y_train, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val.tolist(), y_val, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

    # Modelo e otimizador
    model = GenreClassifier(n_classes=len(mlb.classes_)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)

    # Cálculo de pesos para classes desbalanceadas
    label_sums = y_train.sum(axis=0)
    pos_weights = torch.tensor((len(y_train) - label_sums) / (label_sums + 1e-5)).float().to(device)
    pos_weights = torch.clamp(pos_weights, 0.5, 5.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Loop de treinamento
    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch+1}/{EPOCHS}")
        loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        print(f"Loss: {loss:.4f}")

    # Avaliação
    preds, y_true = eval_model(model, val_loader)
    thresholds = find_best_thresholds(y_true, preds)
    print("Melhores thresholds por classe:", dict(zip(mlb.classes_, thresholds.round(2))))
    get_metrics_with_thresholds(preds, y_true, thresholds, mlb)

# Executa o script
if __name__ == "__main__":
    main()
