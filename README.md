# 🎬 IMDB Movie Genre Classifier (Modalidade Texto)

> Classificação de gêneros de filmes com base em sinopses e posters usando BERT e PyTorch.

![GitHub Repo Size](https://img.shields.io/github/repo-size/seuusuario/imdb-text-classification)
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📚 Sobre o Projeto

Este projeto faz parte da disciplina **Redes Neurais e Aprendizado Profundo** (USP - ICMC) e tem como objetivo classificar os **gêneros cinematográficos** de filmes utilizando **sinopses textuais** e **cartazes** como entrada.

Nesta etapa (modalidade: texto e imagem), utilizamos uma abordagem multimodal que combina representações vetoriais extraídas de sinopses por meio de um modelo pré-treinado BERT (para o texto) com features visuais obtidas a partir de um modelo ResNet50 (para as imagens dos pôsteres). A partir da concatenação dessas duas fontes de informação, o modelo prevê os múltiplos gêneros associados a cada filme.

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

Para esta  entrega, utilizamos  o arquivo:
    IMDB_four_genre_larger_plot_description.csv
    IMDB_four_genre_posters

---

## ⚙️ Como Executar

### 🔧 Instalar Dependências

pip install -r requirements.txt

### 📥 Colocar o Dataset

Crie a pasta data/ e insira o arquivo CSV baixado do Kaggle:

mkdir data
mv IMDB_four_genre_larger_plot_description.csv data/

Insira também a pasta IMDB_four_genre_posters, pois não foi possivel coloca-la dentro do git devido a limite de tamanho. Portanto somente é utilizado o .csv sendo necessario baixar e colocar o arquivo do kaggle

### ▶️ Rodar o Modelo

python train_multilabel_genre_classifier.py --csv_path "path/to/IMDB_four_genre_larger_plot_description.csv" --epochs 8 --batch_size 16 --max_len 256

### ▶️ Modo interativo opcional

Para utilizar o modo interativo, adicione a flag --infer ao comando:

python train_multilabel_genre_classifier.py --csv_path "path/to/ "IMDB_four_genre_larger_plot_description.csv" --img_root path/to \IMDB_four_genre_posters --epochs 8 --batch_size 16 --max_len 256 --infer

Exemplo de Sinopse para Inferência
> Sinopse de Exemplo: > After realizing that her fear of rejection by her parents caused her to hurt Riley and will cause her to lose Abby, Harper finally tells the truth to her parents, confirming that she is a lesbian. Predictably, both Ted and Tipper do not react well to this news. This inspires Sloane to reveal her own secret that she and her husband are getting divorced, and even Jane tells her parents how neglected she felt throughout the years. Harper goes after Abby to apologize, confessing that she truly loves her and wants to build a life with her. Touched by her words, Abby forgives her and they share a long and passionate kiss.

Cole essa sinopse no modo interativo para testar as predições do modelo.

Para testar alguma imagem que não esteja no dataset basta coloca-la dentro de uma das pastas do seu genero respectivo e colocar o path relativo a a ela juntamente com a sua sinopse.


## 🧑‍💻 Autores
Grupo da disciplina SCC0270 — USP - ICMC

Augusto L. Pinto

Gustavo G. Ribeiro

João Francisco CBC de Pinho

Vitor Hugo A. Couto

Eduardo S. Rocha

Antonio Italo Lima Lopes