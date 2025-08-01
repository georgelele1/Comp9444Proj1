import argparse, copy, time, torch, os
from pathlib import Path
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix

DATA_ROOT = Path(r"/Users/invoker/Downloads/Dataset")       # <<< EDIT HERE
NUM_WORKERS = 2                                      # <<< EDIT HERE
PIN_MEMORY  = False                                  # <<< EDIT HERE
IMG_SIZE = 224
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

def get_loaders(batch):
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE,(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tfm_eval = transforms.Compose([
        transforms.Resize(int(IMG_SIZE*1.14)), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    train_ds = datasets.ImageFolder(DATA_ROOT/"train", tfm_train)
    val_ds   = datasets.ImageFolder(DATA_ROOT/"val",   tfm_eval)
    test_ds  = datasets.ImageFolder(DATA_ROOT/"test",  tfm_eval)
    kwargs = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_dl = DataLoader(train_ds, batch, shuffle=True,  **kwargs)
    val_dl   = DataLoader(val_ds,   batch, shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  batch, shuffle=False, **kwargs)
    print(f"Loaded {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test images.")  #
    return train_dl, val_dl, test_dl, train_ds.classes

def build_model(nc, pretrained=True):
    w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = efficientnet_b0(weights=w)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    return m

def run(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", dev)
    tr_dl, va_dl, te_dl, classes = get_loaders(cfg.batch)
    model = build_model(len(classes), not cfg.no_pretrain).to(dev)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs)

    best, best_acc = None, 0
    for ep in range(cfg.epochs):
        t0 = time.time(); model.train()
        tl, tc = 0,0
        for x,y in tr_dl:
            x,y = x.to(dev),y.to(dev); opt.zero_grad()
            out = model(x); loss = loss_fn(out,y); loss.backward(); opt.step()
            tl += loss.item()*x.size(0); tc += out.argmax(1).eq(y).sum().item()
        tl/=len(tr_dl.dataset); tc/=len(tr_dl.dataset)

        vl, vc = 0,0; model.eval()
        with torch.no_grad():
            for x,y in va_dl:
                x,y=x.to(dev),y.to(dev)
                out=model(x); vl+=loss_fn(out,y).item()*x.size(0)
                vc+=out.argmax(1).eq(y).sum().item()
        vl/=len(va_dl.dataset); vc/=len(va_dl.dataset)
        sch.step()
        print(f"Ep {ep+1:02}/{cfg.epochs} tr_loss={tl:.4f} tr_acc={tc:.3f}  val_acc={vc:.3f} ({time.time()-t0:.1f}s)")
        if vc > best_acc:
            best_acc = vc       
            best = model.state_dict().copy()
    model.load_state_dict(best)
    print("Best val acc:", best_acc)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--batch",type=int,default=8); p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--no_pretrain",action="store_true"); cfg=p.parse_args(); run(cfg)
