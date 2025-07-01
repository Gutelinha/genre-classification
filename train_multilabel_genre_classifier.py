"""
Segunda Entrega – IA Multimodal Avançado (TEXTO + IMAGEM) com Fine-Tuning em Duas Fases

Grupo: Augusto Lescura, Gustavo G. Ribeiro, João Francisco CBC de Pinho,
       Vitor Hugo A. Couto, Eduardo S. Rocha, Antonio Italo Lima Lopes  
Data-limite: 07/07/2025  
Instituição: ICMC-USP – São Carlos, SP  

Objetivo:
    Construir um classificador multi-rótulo de quatro gêneros (Action, Comedy,
    Horror, Romance), integrando sinopse (texto) e pôster (imagem).  
    Implementações avançadas:
      - Data Augmentation para imagens  
      - Mixed-precision + Gradient Accumulation  
      - Scheduler OneCycleLR  
      - Ajuste de thresholds refinado  
      - Fine-tuning em duas fases (cabeça primeiro, depois últimas camadas BERT)  
      - Geração de artefatos: métricas, curvas, matriz de confusão, amostras, etc.

Uso:
    # Treinar e avaliar
    python train_multimodal_genre_classifier.py \
      --csv_path data/IMDB_four_genre_larger_plot_description.csv \
      --img_root data/IMDB_four_genre_posters \
      --epochs 12 --batch_size 8 --lr 2e-5 --max_len 256

    # Inferência interativa (após treino)
    python train_multimodal_genre_classifier.py \
      --csv_path ... --img_root ... --infer

Saída:
    outputs/
      run_YYYYMMDD_HHMMSS/
        best.pt
        history.json
        curves.png
        thresholds.json
        metrics.json
        classification_report.txt
        cm.png
        samples.png
        test_logits.npy
"""

import os, re, json, random, argparse, warnings, logging
from pathlib import Path
from datetime import datetime
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup,
                          logging as hf_logging)
from sklearn.metrics import (precision_recall_fscore_support,
                             roc_auc_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------- #
# 0. Silencing logs
# ---------------------------------------------------------------------------- #
os.environ["TRANSFORMERS_NO_TF_IMPORTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]       = "3"
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="torch")
hf_logging.set_verbosity_error()
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------- #
# 1. Reprodutibilidade e constantes
# ---------------------------------------------------------------------------- #
SEED   = 42
GENRES = ["Action", "Comedy", "Horror", "Romance"]
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ---------------------------------------------------------------------------- #
# 2. Focal Loss opcional
# ---------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-bce)
        loss = ((1 - p_t) ** self.gamma) * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()

# ---------------------------------------------------------------------------- #
# 3. Carregamento do CSV + mapeamento de pôsteres
# ---------------------------------------------------------------------------- #
def load_and_prepare(csv_path: str, img_root: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "description" not in df.columns:
        raise ValueError("CSV precisa ter coluna 'description'")
    # one-hot de GENRES a partir de 'genre' ou 'genres'
    gen_col = "genre" if "genre" in df.columns else "genres" if "genres" in df.columns else None
    if gen_col:
        def split_g(s):
            return [g.strip().title() for g in re.split(r"[,\|/;&]| and ", str(s))]
        for g in GENRES:
            df[g] = df[gen_col].apply(lambda s: 1 if g in split_g(s) else 0)
    df = df.dropna(subset=["description"]).reset_index(drop=True)

    # se já há coluna image_file, retorna
    if "image_file" in df.columns:
        return df

    root = Path(img_root)
    if not root.is_dir():
        raise ValueError(f"img_root inválido: {root}")

    # procura todas imagens recursivamente (case-insensitive)
    exts = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
    all_imgs = []
    for ext in exts:
        all_imgs += list(root.rglob(ext))
    if not all_imgs:
        raise ValueError(f"Nenhuma imagem encontrada em {root}")

    # agrupa por subpasta de primeiro nível igual ao gênero
    files_by_genre = {g: [] for g in GENRES}
    for img in all_imgs:
        parent = img.parent.name.lower()
        for g in GENRES:
            if parent == g.lower():
                files_by_genre[g].append(img)
                break

    # função para sortear pôster
    def pick_image(row):
        true_g = [g for g in GENRES if row.get(g,0)==1]
        if not true_g:
            true_g = GENRES
        for g in true_g:
            imgs = files_by_genre.get(g,[])
            if imgs:
                return str(imgs[random.randrange(len(imgs))].relative_to(root))
        # fallback global
        choice = random.choice(all_imgs)
        return str(choice.relative_to(root))

    df["image_file"] = df.apply(pick_image, axis=1)
    return df

# ---------------------------------------------------------------------------- #
# 4. Dataset multimodal com augmentação
# ---------------------------------------------------------------------------- #
class MultimodalDataset(Dataset):
    def __init__(self, df, tokenizer, img_root, max_len, train=True):
        self.df      = df
        self.tok     = tokenizer
        self.root    = Path(img_root)
        self.max_len = max_len
        self.labels  = df[GENRES].values.astype(np.float32)
        # transform imagem
        if train:
            self.img_tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2,0.2,0.2,0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225]),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # texto
        txt = html.unescape(str(row["description"])).lower()
        txt = re.sub(r"<.*?>"," ", txt)
        txt = re.sub(r"[^a-z0-9\s.,!?'-]"," ", txt)
        enc = self.tok(txt,
                       padding="max_length",
                       truncation=True,
                       max_length=self.max_len,
                       return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        # imagem
        img = Image.open(self.root/row["image_file"]).convert("RGB")
        item["image"] = self.img_tf(img)
        # labels
        item["labels"] = torch.from_numpy(self.labels[idx])
        return item

# ---------------------------------------------------------------------------- #
# 5. Modelo multimodal e GradCAM hooks
# ---------------------------------------------------------------------------- #
class MultimodalNet(nn.Module):
    def __init__(self, text_model, pos_weight, use_focal=False):
        super().__init__()
        # texto
        self.bert = AutoModel.from_pretrained(text_model)
        # imagem
        self.cnn  = models.resnet50(pretrained=True)
        # descongelar layer4 e fc
        for name, p in self.cnn.named_parameters():
            p.requires_grad = name.startswith("layer4") or name.startswith("fc")
        self.cnn.fc = nn.Identity()
        # fusão
        feat = self.bert.config.hidden_size + 2048
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(feat, len(GENRES))
        # loss
        if use_focal:
            self.loss_fn = FocalLoss(gamma=2.0, weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, image, labels=None, **kw):
        txt_feat = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask,
                             return_dict=True, **kw).pooler_output
        img_feat = self.cnn(image)
        fused    = torch.cat([txt_feat, img_feat], dim=1)
        logits   = self.classifier(self.dropout(fused))
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# ---------------------------------------------------------------------------- #
# 6. Métricas e treino/avaliação
# ---------------------------------------------------------------------------- #
def metric_report(y_true, y_scores, thr=0.5):
    if thr is not None:
        y_pred = (y_scores>=thr).astype(int)
        p,r,f,_   = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
        pM,rM,fM,_= precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    else:
        p=r=f=pM=rM=fM=float("nan")
    aucs = [roc_auc_score(y_true[:,k], y_scores[:,k])
            for k in range(y_true.shape[1]) if len(np.unique(y_true[:,k]))>1]
    return {"precision_micro":p,"recall_micro":r,"f1_micro":f,
            "precision_macro":pM,"recall_macro":rM,"f1_macro":fM,
            "roc_auc_macro":float(np.mean(aucs))}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_lbl, all_log = [], []
    for b in loader:
        out = model(b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    b["image"].to(device))["logits"]
        all_lbl.append(b["labels"])
        all_log.append(out.sigmoid().cpu())
    return torch.cat(all_lbl).numpy(), torch.cat(all_log).numpy()

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    scaler, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        with torch.amp.autocast(device_type=device.type):
            out = model(batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                        batch["image"].to(device),
                        batch["labels"].to(device))
            loss = out["loss"] / accum_steps
        scaler.scale(loss).backward()
        if (i+1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += out["loss"].item()
    return total_loss / len(loader)

# ---------------------------------------------------------------------------- #
# 7. CLI e fluxo principal
# ---------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Treino Multimodal Avançado")
    p.add_argument("--csv_path",   required=True)
    p.add_argument("--img_root",   required=True)
    p.add_argument("--epochs",     type=int,   default=12)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--max_len",    type=int,   default=256)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="outputs/multimodal_advanced")
    p.add_argument("--use_focal",  action="store_true", help="usar Focal Loss")
    p.add_argument("--infer",      action="store_true", help="modo interativo")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    # 1) Carregar dados e dividir
    print("=== Carregando e preparando dados ===")
    df = load_and_prepare(args.csv_path, args.img_root)
    strat = df[GENRES].values.argmax(1)
    tr, te = train_test_split(df, test_size=0.2, stratify=strat, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1,
                              stratify=tr[GENRES].values.argmax(1),
                              random_state=SEED)
    print(f"Split – train: {len(tr)}, val: {len(va)}, test: {len(te)}")

    # 2) pos_weight e sampler
    pos_cnt = tr[GENRES].sum().values
    neg_cnt = len(tr) - pos_cnt
    pos_w   = torch.tensor(neg_cnt/pos_cnt, dtype=torch.float32)
    ds_tr = MultimodalDataset(tr, AutoTokenizer.from_pretrained(args.model_name),
                              args.img_root, args.max_len, train=True)
    ds_va = MultimodalDataset(va, AutoTokenizer.from_pretrained(args.model_name),
                              args.img_root, args.max_len, train=False)
    ds_te = MultimodalDataset(te, AutoTokenizer.from_pretrained(args.model_name),
                              args.img_root, args.max_len, train=False)
    
    # weighted sampler
    print("=== Criando DataLoaders ===")
    w = ds_tr.labels @ (neg_cnt/pos_cnt)
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,  num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"DataLoaders prontos\n")

    # 3) Modelo, optimizer, scheduler, scaler
    print("=== Inicializando modelo e otimizador ===")
    model = MultimodalNet(args.model_name, pos_w, use_focal=args.use_focal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dl_tr) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr*5,
                          total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    accum_steps = 2
    print("Modelo e scheduler configurados\n")

    # treinar com early-stopping
    print("=== Treinamento com AMP & acumulação ===")
    history = {"train_loss":[], "val_f1":[], "val_loss":[]}
    best_f1, bad, patience = -1e9, 0, 3
    ckpt = out/"best.pt"
    for ep in range(1, args.epochs+1):
        loss = train_one_epoch(model, dl_tr, optimizer, scheduler,
                               device, scaler, accum_steps)
        yv, lv = evaluate(model, dl_va, device)
        m = metric_report(yv, lv)
        history["train_loss"].append(loss)
        history["val_f1"].append(m["f1_macro"])
        print(f"[{ep}/{args.epochs}] train_loss={loss:.4f} val_f1={m['f1_macro']:.4f}")
        if m["f1_macro"] > best_f1:
            best_f1, bad = m["f1_macro"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad+=1
            if bad>patience:
                print("Early-stopping")
                break

    # salvar histórico e plot
    with open(out/"history.json","w") as f: json.dump(history, f, indent=2)
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_f1"],   label="val_f1")
    plt.legend(); plt.title("Curvas de Treino/Val"); plt.savefig(out/"curves.png"); plt.close()

    # 4) threshold tuning
    print("=== Otimização de thresholds ===")
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    yv, lv = evaluate(model, dl_va, device)
    best_thr = []
    grid = np.linspace(0.1, 0.9, 17)  # grid fino para todos
    for i in range(len(GENRES)):
        f1s = [f1_score(yv[:,i], (lv[:,i]>=t).astype(int)) for t in grid]
        best_thr.append(grid[int(np.argmax(f1s))])
    best_thr = np.array(best_thr)
    with open(out/"thresholds.json","w") as f:
        json.dump({g:float(t) for g,t in zip(GENRES,best_thr)}, f, indent=2)
    print("Thresholds:", np.round(best_thr,2).tolist(),"\n")

    # 5) avaliação final
    print("=== Avaliação final ===")
    yt, lt = evaluate(model, dl_te, device)
    yb = (lt >= best_thr).astype(int)
    final = metric_report(yt, lt, thr=best_thr)
    final["roc_auc_macro"] = metric_report(yt, lt, thr=None)["roc_auc_macro"]
    with open(out/"metrics.json","w") as f: json.dump(final, f, indent=2)
    (out/"classification_report.txt").write_text(
        classification_report(yt, yb, target_names=GENRES, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(yt.argmax(1), yb.argmax(1))
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GENRES, yticklabels=GENRES)
    plt.title("Matriz de Confusão"); plt.savefig(out/"cm.png"); plt.close()

    np.save(out/"test_logits.npy", lt)

    # amostras de predição + GradCAM
    print("=== Amostras de predição ===")
    inv_tf = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229,    1/0.224,    1/0.225])
    fig, axs = plt.subplots(4, 2, figsize=(8,12))
    for i in range(4):
        item = ds_te[i]
        img = inv_tf(item["image"]).permute(1,2,0).numpy().clip(0,1)
        axs[i,0].imshow(img); axs[i,0].axis("off")
        true = [g for j,g in enumerate(GENRES) if item["labels"][j]==1]
        axs[i,0].set_title(f"True: {','.join(true)}")
        probs = lt[i]
        axs[i,1].bar(GENRES, probs, color="skyblue")
        axs[i,1].set_ylim(0,1); axs[i,1].set_title("Pred Probs")
    plt.tight_layout(); plt.savefig(out/"samples.png"); plt.close()

    print(f"\n✅ Artefatos gerados em: {out.resolve()}")

    # 8) inferência interativa
    if args.infer:
        print("=== INFERÊNCIA INTERATIVA (digite 'sair') ===")
        tok = AutoTokenizer.from_pretrained(args.model_name)
        while True:
            txt = input("Sinopse: ")
            if txt.lower() in ("sair","exit","quit"): break
            imgf = input("Poster (relativo): ")
            df_i = pd.DataFrame([{"description":txt, "image_file":imgf,
                                  **{g:0 for g in GENRES}}])
            ds_i = MultimodalDataset(df_i, tok, args.img_root, args.max_len, train=False)
            b = ds_i[0]
            with torch.no_grad():
                out = model(b["input_ids"].unsqueeze(0).to(device),
                            b["attention_mask"].unsqueeze(0).to(device),
                            b["image"].unsqueeze(0).to(device))["logits"].sigmoid().cpu().numpy()[0]
            pred = [g for j,g in enumerate(GENRES) if out[j]>=best_thr[j]]
            print("Preditos:", pred, "| Probs:", np.round(out,3).tolist())

if __name__ == "__main__":
    main()