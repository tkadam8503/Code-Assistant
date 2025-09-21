import argparse, os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50, densenet121
import medmnist
from medmnist import OCTMNIST
from tqdm import tqdm

def get_model(backbone, num_classes):
    if backbone == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif backbone == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError("Unknown backbone")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet50","densenet121"], default="resnet50")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="out_model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    info = medmnist.INFO["octmnist"]
    num_classes = len(info["label"])

    transform = T.Compose([
        T.ToTensor(),
        T.RandomApply([T.GaussianBlur(3)], p=0.3),
        T.RandomApply([T.Lambda(lambda x: x + 0.05*torch.randn_like(x))], p=0.5),
        T.Normalize(mean=[.5], std=[.5])
    ])

    train_ds = OCTMNIST(split="train", transform=transform, download=True, as_rgb=False)
    val_ds = OCTMNIST(split="val", transform=T.Compose([T.ToTensor(), T.Normalize([.5],[.5])]), download=True, as_rgb=False)

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.backbone, num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Train ep{ep+1}"):
            x, y = x.to(device), y.squeeze().long().to(device)
            opt.zero_grad()
            logits = model(x.repeat(1,3,1,1))  # expand to 3 channels
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.squeeze().long().to(device)
                logits = model(x.repeat(1,3,1,1))
                pred = logits.argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        acc = correct/total if total>0 else 0
        print(f"Val acc: {acc:.3f}")

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        f.write(json.dumps({"backbone": args.backbone, "num_classes": num_classes}))

if __name__ == "__main__":
    import json
    main()
