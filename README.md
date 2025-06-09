# 🎬 IMDB Movie Genre Classifier (Modalidade Texto)

> Classificação de gêneros de filmes com base em sinopses usando BERT e PyTorch.

![GitHub Repo Size](https://img.shields.io/github/repo-size/seuusuario/imdb-text-classification)
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📚 Sobre o Projeto

Este projeto faz parte da disciplina **Redes Neurais e Aprendizado Profundo** (USP - ICMC) e tem como objetivo classificar os **gêneros cinematográficos** de filmes utilizando **sinopses textuais** como entrada.

Nesta etapa (**modalidade: texto**), usamos um modelo pré-treinado `BERT` para gerar representações vetoriais das sinopses e, a partir disso, prever múltiplos gêneros associados ao filme.

---

## 🧠 Tecnologias Utilizadas

- [Python 3.8+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Transformers (HuggingFace)](https://huggingface.co/transformers/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Pandas, NumPy, tqdm, Matplotlib](https://pandas.pydata.org/)

---

## 🗃️ Dataset

Utilizamos o dataset multimodal do Kaggle:

🎯 [IMDB Multimodal Vision and NLP Genre Classification](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-multimodal-vision-and-nlp-genre-classification)

Para esta primeira entrega, utilizamos apenas o arquivo:
    IMDB_four_genre_larger_plot_description.csv

---

## ⚙️ Como Executar

### 🔧 Instalar Dependências

pip install -r requirements.txt

### 📥 Colocar o Dataset

Crie a pasta data/ e insira o arquivo CSV baixado do Kaggle:

mkdir data
mv IMDB_four_genre_larger_plot_description.csv data/

### ▶️ Rodar o Modelo

python main.py

---

## 🚀 Próximos Passos

✅ Modalidade Texto (sinopse com BERT)

🔜 Modalidade Imagem (cartaz com CNN)

🔜 Classificação Multimodal (fusão texto + imagem)

---

## 🧑‍💻 Autores
Grupo da disciplina SCC0270 — USP - ICMC

Augusto L. Pinto

Gustavo G. Ribeiro

João Francisco CBC de Pinho

Vitor Hugo A. Couto

Eduardo S. Rocha

Antonio Italo Lima Lopes