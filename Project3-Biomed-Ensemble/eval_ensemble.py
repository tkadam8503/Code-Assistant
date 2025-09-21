import argparse, os, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50, densenet121
import medmnist
from medmnist import OCTMNIST
from tqdm import tqdm
import torch.nn.functional as F

def load_model(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    bb = meta["backbone"]
    num_classes = meta["num_classes"]
    if bb == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    m.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location="cpu"))
    m.eval()
    return m, bb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    args = ap.parse_args()

    transform = T.Compose([T.ToTensor(), T.Normalize([.5],[.5])])
    test_ds = OCTMNIST(split="test", transform=transform, download=True, as_rgb=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in args.paths:
        m, bb = load_model(p)
        models.append(m.to(device))
        print("Loaded", bb, "from", p)

    correct_ens = total_ens = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device), y.squeeze().long().to(device)
            probs = []
            for m in models:
                logits = m(x.repeat(1,3,1,1))
                probs.append(F.softmax(logits, dim=1))
            stacked = torch.stack(probs)  # m, b, c
            max_conf = stacked.max(dim=2).values  # m, b
            weights = max_conf / (max_conf.sum(dim=0, keepdim=True)+1e-9)  # m, b
            weighted = (stacked * weights.unsqueeze(2)).sum(dim=0)  # b, c
            pred_ens = weighted.argmax(1)
            correct_ens += (pred_ens==y).sum().item()
            total_ens += y.numel()

    print(f"Ensemble accuracy: {correct_ens/total_ens:.3f}")

if __name__ == "__main__":
    main()
