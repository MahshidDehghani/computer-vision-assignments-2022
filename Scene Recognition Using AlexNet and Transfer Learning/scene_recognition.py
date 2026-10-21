import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_loaders(data_dir, batch_size, num_workers, resize, crop, augment=True):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = [transforms.Resize((resize, resize))]
    if augment:
        train_tf += [
            transforms.RandomCrop((crop, crop)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        train_tf += [transforms.CenterCrop((crop, crop))]
    train_tf += [transforms.ToTensor(), norm]
    test_tf = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop((crop, crop)),
        transforms.ToTensor(), norm,
    ])
    train_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "Train"),
                                                 transforms.Compose(train_tf))
    test_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "Test"), test_tf)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)
    # un-augmented loader on the training data, used to measure train loss/accuracy
    train_eval_set = torchvision.datasets.ImageFolder(os.path.join(data_dir, "Train"), test_tf)
    train_eval_loader = DataLoader(train_eval_set, shuffle=False, **kw)
    return train_loader, train_eval_loader, test_loader, len(train_set.classes)


class AlexNet1(nn.Module):
    """First conv layer + last two FC layers of AlexNet (4x4 max pool, stride 4)."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


class AlexNet2(nn.Module):
    """First three conv layers (256 filters in conv3 + 2x2/2 pool) + 3 FC layers."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def build_model(exp, num_classes):
    if exp == 1:
        return AlexNet1(num_classes)
    if exp == 2:
        return AlexNet2(num_classes)
    if exp == 3:  # full AlexNet, random init
        return torchvision.models.alexnet(weights=None, num_classes=num_classes)
    # exp 4 / 5: ImageNet-pretrained AlexNet, new last layer
    try:
        model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    except AttributeError:  # older torchvision
        model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)  # random init
    if exp == 4:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier[6].parameters():
            p.requires_grad = True
    return model


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, k5):
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct = out.topk(k5, dim=1).indices == y.unsqueeze(1)
        top1 += correct[:, 0].sum().item()
        top5 += correct.any(dim=1).sum().item()
        n += y.size(0)
    return loss_sum / n, 100.0 * top1 / n, 100.0 * top5 / n


def run_experiment(exp, args):
    print(f"\n{'=' * 60}\nExperiment {exp}\n{'=' * 60}")
    pretrained = exp in (4, 5)
    resize, crop = (256, 224) if pretrained else (args.scratch_resize, args.scratch_crop)
    train_loader, train_eval_loader, test_loader, num_classes = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, resize, crop,
        augment=not args.no_augment)
    k5 = min(5, num_classes)

    model = build_model(exp, num_classes).to(DEVICE)
    lr = {1: args.lr, 2: args.lr, 3: args.lr, 4: args.lr_frozen, 5: args.lr_finetune}[exp]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=4)

    hist = {k: [] for k in ["tr_loss", "te_loss", "tr_t1", "te_t1", "tr_t5", "te_t5"]}

    def log(epoch):
        trl, tr1, tr5 = evaluate(model, train_eval_loader, k5)
        tel, te1, te5 = evaluate(model, test_loader, k5)
        for key, v in zip(hist, [trl, tel, tr1, te1, tr5, te5]):
            hist[key].append(v)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss tr {trl:.4f} te {tel:.4f} "
              f"| top1 tr {tr1:.2f} te {te1:.2f} | top5 tr {tr5:.2f} te {te5:.2f}")
        return tel

    ckpt = f"model_p{exp}.pt"
    start = time.time()
    best_loss = log(0)  # epoch 0 = initial weights
    torch.save(model.state_dict(), ckpt)
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        te_loss = log(epoch)
        scheduler.step(te_loss)
        if te_loss < best_loss:
            best_loss, best_epoch = te_loss, epoch
            torch.save(model.state_dict(), ckpt)
    print(f"Done in {(time.time() - start) / 60:.1f} min. "
          f"Best test loss {best_loss:.4f} at epoch {best_epoch}")

    plot(hist["tr_loss"], hist["te_loss"], "Loss", f"p{exp}_loss.png", exp)
    plot(hist["tr_t1"], hist["te_t1"], "Top-1 accuracy (%)", f"p{exp}_top1.png", exp)
    plot(hist["tr_t5"], hist["te_t5"], f"Top-{k5} accuracy (%)", f"p{exp}_top5.png", exp)

    return {
        "best_epoch_by_test_loss": best_epoch,
        "best_test_loss": best_loss,
        "test_top1_at_best_loss": hist["te_t1"][best_epoch],
        "test_top5_at_best_loss": hist["te_t5"][best_epoch],
        "max_test_top1": max(hist["te_t1"]),
        "max_test_top5": max(hist["te_t5"]),
    }


def plot(train_vals, test_vals, ylabel, fname, exp):
    plt.figure()
    epochs = range(len(train_vals))  # includes epoch 0 (initial weights)
    plt.plot(epochs, train_vals, marker="o", markersize=3, label="Train")
    plt.plot(epochs, test_vals, marker="o", markersize=3, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Experiment {exp}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    choices=[1, 2, 3, 4, 5])
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01, help="exp 1-3 (random init)")
    ap.add_argument("--lr_frozen", type=float, default=0.01, help="exp 4 (last layer only)")
    ap.add_argument("--lr_finetune", type=float, default=0.001, help="exp 5 (fine tuning)")
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--scratch_resize", type=int, default=128, help="input resize, exp 1-3")
    ap.add_argument("--scratch_crop", type=int, default=112, help="input crop, exp 1-3")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print("Device:", DEVICE)

    results = {}
    for exp in args.exp:
        results[f"exp{exp}"] = run_experiment(exp, args)
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\nSummary")
    for name, r in results.items():
        print(f"{name}: best test top1 {r['max_test_top1']:.2f}% | "
              f"top5 {r['max_test_top5']:.2f}% | min test loss {r['best_test_loss']:.4f} "
              f"(epoch {r['best_epoch_by_test_loss']})")

