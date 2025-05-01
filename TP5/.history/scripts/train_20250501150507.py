import os
import torch
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from scripts.helpers import get_datasets
from tqdm import tqdm
import datetime
import argparse
import pandas as pd

def train(
    epochs: int,
    num_classes: int,
    batch_size: int,
    warm_up: int = 4,          # Phase 1: head-only
    full_unfreeze: int = 8,   # Phase 3: unfreeze everything
    base_lr: float = 3e-4,
    head_lr: float = 3e-3,
    clip_norm: float = 1.0
):
    device = get_device()
    print(f"Using device: {device}")

    model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=num_classes).to(device)

    # Phase/parameter group split
    head_params, late_params, early_params = [], [], []
    late_block_ids = {id(p) for m in model._blocks[-2:] for p in m.parameters()}
    late_block_ids |= {id(p) for p in model._conv_head.parameters()}

    for n, p in model.named_parameters():
        if "_fc" in n:
            head_params.append(p)
        elif id(p) in late_block_ids:
            late_params.append(p)
            p.requires_grad_(False)
        else:
            early_params.append(p)
            p.requires_grad_(False)

    optimizer = optim.AdamW(
        [
            {"params": early_params, "lr": 0.0},
            {"params": late_params,  "lr": 0.0},
            {"params": head_params,  "lr": head_lr},
        ],
        weight_decay=1e-3
    )

    counts = np.array([4522, 12875, 3323, 867, 2624, 239, 253, 628], dtype=np.float32)
    w = 1.0 / np.sqrt(counts)
    w /= w.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))

    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    train_dataset, val_dataset = get_datasets(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_state, best_acc = None, 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        if epoch == warm_up:
            for p in late_params:
                p.requires_grad_(True)
            optimizer.param_groups[1]["lr"] = base_lr
        if epoch == full_unfreeze:
            for p in early_params:
                p.requires_grad_(True)
            optimizer.param_groups[0]["lr"] = base_lr * 0.3

        model.train()
        total_loss, correct = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            y_labels = y.argmax(dim=1)
            loss = criterion(out, y_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y_labels).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)

        model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                y_labels = y.argmax(dim=1)
                loss = criterion(out, y_labels)
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y_labels).sum().item()
        val_loss = total_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return train_losses, val_losses, train_accs, val_accs, best_state



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--data-dir", type=str, default="data/skin-lesion-fake")
    parser.add_argument("--output-dir", type=str, default="models/saved_models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_losses, val_losses, train_accs, val_accs, best_model = train(
        args.epochs, args.num_classes, args.batch_size
    )

    date = datetime.date.today().isoformat()
    model_path = os.path.join(args.output_dir, f"{date}_model.pth")
    torch.save(best_model, model_path)

    for name, values in [("train_loss", train_losses), ("val_loss", val_losses), ("train_acc", train_accs), ("val_acc", val_accs)]:
        df = pd.DataFrame({name: values})
        df.to_csv(os.path.join(args.output_dir, f"{date}_{name}.csv"), index=False)
    print("Saved model to", model_path)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
