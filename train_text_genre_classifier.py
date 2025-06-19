"""
Trabalho Prático – IA Multimodal (ENTREGA 1)
Grupo: Augusto Lescura, Gustavo G. Ribeiro, João Francisco CBC de Pinho, Vitor Hugo A. Couto,
Eduardo S. Rocha e Antonio Italo Lima Lopes.
Data-limite: 22/06/2025
Instituição: ICMC-USP – São Carlos – SP – Brasil

Problema
========
Classificar cada filme do conjunto “IMDB Multimodal – Vision & NLP Genre Classification”
(em CSV obtido do Kaggle) em quatro gêneros (multi-label):

    Action | Comedy | Horror | Romance

Nesta primeira entrega utilizamos apenas a modalidade TEXTO da sinopse (campo
"description"), realizando pré-processamento, divisão treino/val/teste e comparando:

1. Baseline: TF-IDF + LogisticRegression (One-vs-Rest)
2. Modelo principal: BERT-base uncased com fine-tuning

Melhorias incluídas:
--------------------
∙ Limpeza textual simples (remoção de tags HTML, datas, símbolos indesejados)  
∙ Uso de BCEWithLogitsLoss com pos_weight para rótulos desequilibrados  
∙ WeightedRandomSampler para oversampling dos rótulos minoritários  
∙ Early-Stopping com paciência = 2  
∙ Grid search para encontrar o melhor threshold por rótulo  
∙ Modo interativo (--infer): insira uma sinopse e obtenha os gêneros preditos

Requisitos:
-----------
torch>=2.2, transformers, scikit-learn, pandas, numpy, matplotlib, seaborn

Modo de uso:
------------
Para treinar e depois utilizar o modo interativo, execute:

    python train_text_genre_classifier.py --csv_path "data/IMDB_four_genre_larger_plot_description.csv" --epochs 8 --batch_size 16 --max_len 256 --infer
================================================================================
"""
# -----------------------------------------------------------------------------#
# 0. IMPORTS + SILENCIAMENTO DE LOGS
# -----------------------------------------------------------------------------#
from __future__ import annotations
import argparse, random, re, html, warnings, logging, os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             classification_report, f1_score, confusion_matrix)
import matplotlib.pyplot as plt, seaborn as sns
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup,
                          logging as hf_logging)

# Silenciar avisos/reports desnecessários
os.environ["TRANSFORMERS_NO_TF_IMPORTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning,  module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
hf_logging.set_verbosity_error()
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------#
# 1. REPRODUTIBILIDADE
# -----------------------------------------------------------------------------#
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

GENRES = ["Action", "Comedy", "Horror", "Romance"]

# -----------------------------------------------------------------------------#
# 2. DATASET – Classe PyTorch
# -----------------------------------------------------------------------------#
class IMDBTextDataset(Dataset):
    """
    Dataset que retorna os inputs para BERT (input_ids, attention_mask, labels).
    Aplica uma limpeza básica no texto para remover ruídos.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["plot_description"].apply(self._clean).tolist()
        self.labels = df[GENRES].values.astype(np.float32)
        self.tokenizer, self.max_len = tokenizer, max_len

    @staticmethod
    def _clean(t: str) -> str:
        t = html.unescape(str(t)).lower()
        t = re.sub(r"<.*?>", " ", t)        # Remove tags HTML
        t = re.sub(r"\d{4}", " ", t)         # Remove anos (ex.: 1999)
        t = re.sub(r"[^a-z0-9\s.,!?'-]", " ", t)
        return re.sub(r"\s{2,}", " ", t).strip()

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx],
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_len,
                             return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.from_numpy(self.labels[idx])
        return item

# -----------------------------------------------------------------------------#
# 3. MODELO – BERT + Head Linear multi-label
# -----------------------------------------------------------------------------#
class BertForMultiLabel(nn.Module):
    """
    Modelo BERT fine-tunado com head linear para multi-label.
    Utiliza BCEWithLogitsLoss com ponderação para lidar com o desbalanceamento.
    """
    def __init__(self, model_name: str, pos_weight: torch.Tensor):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(GENRES))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Aceita parâmetros extras (como token_type_ids) que serão passados para o BERT.
        pooled = self.bert(input_ids,
                           attention_mask=attention_mask,
                           return_dict=True,
                           **kwargs).pooler_output
        logits = self.classifier(self.dropout(pooled))
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# -----------------------------------------------------------------------------#
# 4. MÉTRICAS AUXILIARES
# -----------------------------------------------------------------------------#
def metric_report(y_true: np.ndarray, y_scores: np.ndarray, thr: float | None = 0.5) -> dict:
    """
    Calcula métricas via threshold.
    Se thr for None, calcula apenas o ROC-AUC macro (usado pra otimizar threshold).
    """
    if thr is not None:
        y_pred = (y_scores >= thr).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0)
        pM, rM, fM, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
    else:
        p = r = f = pM = rM = fM = float("nan")
    aucs = [roc_auc_score(y_true[:, k], y_scores[:, k])
            for k in range(y_true.shape[1]) if len(np.unique(y_true[:, k])) > 1]
    auc_macro = float(np.mean(aucs)) if aucs else float("nan")
    return {"precision_micro": p, "recall_micro": r, "f1_micro": f,
            "precision_macro": pM, "recall_macro": rM, "f1_macro": fM,
            "roc_auc_macro": auc_macro}

# -----------------------------------------------------------------------------#
# 5. FUNÇÕES DE TREINO / AVALIAÇÃO
# -----------------------------------------------------------------------------#
def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["labels"].to(device))
        outputs["loss"].backward()
        optimizer.step()
        scheduler.step()
        total_loss += outputs["loss"].item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device),
                       batch["attention_mask"].to(device))["logits"]
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
    return (torch.cat(all_labels).numpy(),
            torch.cat(all_logits).sigmoid().numpy())

# -----------------------------------------------------------------------------#
# 6. FUNÇÃO PARA PREVISÃO INTERATIVA
# -----------------------------------------------------------------------------#
def predict_synopsis(model, tokenizer, text: str, device, threshold: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Recebe uma sinopse (texto), retorna os gêneros preditos e as probabilidades.
    Aplica a mesma limpeza utilizada no treinamento.
    """
    # Limpeza igual à usada no dataset:
    text_clean = IMDBTextDataset._clean(text)
    enc = tokenizer(text_clean, padding="max_length",
                    truncation=True, max_length=256,
                    return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**enc)["logits"].sigmoid().cpu().numpy()[0]
    pred = (logits >= threshold).astype(int)
    predicted_genres = [GENRES[i] for i, p in enumerate(pred) if p == 1]
    return predicted_genres, logits

# -----------------------------------------------------------------------------#
# 7. FUNÇÃO DE CARREGAMENTO DO CSV E PREPARAÇÃO (One-Hot)
# -----------------------------------------------------------------------------#
def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "description" not in df.columns:
        raise ValueError("CSV precisa ter coluna 'description'.")
    genre_col = "genre" if "genre" in df.columns else "genres"
    if genre_col not in df.columns:
        raise ValueError("CSV não contém coluna 'genre(s)'.")
    def split(s): return [p.strip().lower() for p in re.split(r"[,\|/;&]| and ", str(s))]
    def to_hot(s):
        tokens = set(split(s))
        return pd.Series([1 if g.lower() in tokens else 0 for g in GENRES])
    df[GENRES] = df[genre_col].apply(to_hot)
    return (df[["description", *GENRES]]
            .rename(columns={"description": "plot_description"})
            .dropna(subset=["plot_description"]))

# -----------------------------------------------------------------------------#
# 8. ARGUMENTOS CLI
# -----------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(
        description="ENTREGA 1 – Classificação multi-label de gênero via sinopse")
    p.add_argument("--csv_path", required=True,
                   help="Caminho para o CSV do Kaggle.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="outputs/")
    p.add_argument("--infer", action="store_true",
                   help="Ativa modo interativo de inferência após treinamento.")
    return p.parse_args()

# -----------------------------------------------------------------------------#
# 9. MAIN – TREINO, AVALIAÇÃO E (OPCIONAL) INFERÊNCIA INTERATIVA
# -----------------------------------------------------------------------------#
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 9.1 Carrega DataFrame e divide em treino/val/test ---
    df = load_and_prepare(args.csv_path)
    tr, te = train_test_split(df, test_size=0.2,
                              stratify=df[GENRES].values.argmax(1),
                              random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1,
                              stratify=tr[GENRES].values.argmax(1),
                              random_state=SEED)

    # --- 9.2 Baseline TF-IDF + LogisticRegression (OvR) ---
    print("\n===== Baseline TF-IDF + LogisticRegression (OvR) =====")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words="english")
    X_tr, X_te = tfidf.fit_transform(tr["plot_description"]), tfidf.transform(te["plot_description"])
    y_tr, y_te = tr[GENRES].values, te[GENRES].values
    ovr = OneVsRestClassifier(LogisticRegression(max_iter=500, n_jobs=-1))
    ovr.fit(X_tr, y_tr)
    print("Baseline:", metric_report(y_te, ovr.predict_proba(X_te)))

    # --- 9.3 Preparação para BERT ---
    pos_counts = tr[GENRES].sum().values
    neg_counts = len(tr) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    print("Pos_weight:", pos_weight.tolist())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds_tr = IMDBTextDataset(tr, tokenizer, args.max_len)
    ds_va = IMDBTextDataset(va, tokenizer, args.max_len)
    ds_te = IMDBTextDataset(te, tokenizer, args.max_len)

    sample_weights = (ds_tr.labels @ (neg_counts / pos_counts))
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 9.4 Define modelo, optimizer, scheduler ---
    model = BertForMultiLabel(args.model_name, pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dl_tr) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # --- 9.5 Treinamento com Early-Stopping ---
    best_f1, patience, bad = -1e9, 2, 0
    ckpt = out_dir / "best.pt"
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, dl_tr, optimizer, scheduler, device)
        y_va, va_logits = evaluate(model, dl_va, device)
        m_val = metric_report(y_va, va_logits)
        print(f"[{epoch}/{args.epochs}] loss={loss:.4f}  val_f1_macro={m_val['f1_macro']:.4f}")
        if m_val["f1_macro"] > best_f1:
            best_f1, bad = m_val["f1_macro"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad > patience:
                print("Early-stopping 🚦")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True))

    # --- 9.6 Otimização do threshold por rótulo ---
    y_va, va_logits = evaluate(model, dl_va, device)
    best_thr = []
    for k in range(len(GENRES)):
        grid = np.linspace(0.1, 0.9, 9)
        f1s = [f1_score(y_va[:, k], (va_logits[:, k] >= t).astype(int))
               for t in grid]
        best_thr.append(grid[int(np.argmax(f1s))])
    best_thr = np.array(best_thr)
    print("Thresholds ótimos:", np.round(best_thr, 2).tolist())

    # --- 9.7 Avaliação final ---
    y_test, test_logits = evaluate(model, dl_te, device)
    y_bin = (test_logits >= best_thr).astype(int)
    test_metrics = metric_report(y_test, test_logits, thr=None)
    print("\n=== TEST ROC-AUC macro ===", test_metrics["roc_auc_macro"])
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_bin, target_names=GENRES, zero_division=0))
    cm = confusion_matrix(y_test.argmax(1), y_bin.argmax(1))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GENRES, yticklabels=GENRES)
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png"); plt.close()
    np.save(out_dir / "test_logits.npy", test_logits)
    print(f"\n✅ Artefatos salvos em: {out_dir.resolve()}")

    # --- 9.8 Modo interativo de inferência (se --infer for especificado) ---
    if args.infer:
        print("\n============ MODO INTERATIVO DE INFERÊNCIA ============")
        print("Digite 'sair' para terminar.")
        while True:
            user_input = input("\nDigite uma sinopse: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break
            predicted_genres, probs = predict_synopsis(model, tokenizer, user_input, device, best_thr)
            print("Gêneros preditos:", predicted_genres)
            print("Probabilidades:", np.round(probs, 3).tolist())

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
