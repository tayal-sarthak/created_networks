import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from KANConv import KAN_Convolutional_Layer


# simple cifar kan runner
# mps if available; small batch, no workers, 6h


class KANC_CIFAR(nn.Module):
    def __init__(self, grid_size: int = 3, num_classes: int = 100):
        super().__init__()
        # block 1: 3 -> 8
        self.conv1 = KAN_Convolutional_Layer(
            in_channels=3,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # 32 -> 16

        # block 2: 8 -> 16
        self.conv2 = KAN_Convolutional_Layer(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # 16 -> 8

        # block 3: 16 -> 32
        self.conv3 = KAN_Convolutional_Layer(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            grid_size=grid_size,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # 8 -> 4

        # flatten and linear
        self.flat = nn.Flatten()
        with torch.no_grad():
            d = torch.zeros(1, 3, 32, 32)
            y = self.pool1(self.conv1(d))
            y = self.pool2(self.conv2(y))
            y = self.pool3(self.conv3(y))
            feat_dim = y.numel()
        self.linear = nn.Linear(feat_dim, num_classes)

        self.name = f"kanc cifar (3 blocks, gs={grid_size})"

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flat(x)
        x = self.linear(x)  # logits
        return x


def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f"epoch {epoch} [train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=float(loss))


def test(model, device, test_loader, loss_fn):
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[test]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss_sum += loss_fn(logits, labels).item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
    avg_loss = loss_sum / len(test_loader)
    n = len(test_loader.dataset)
    acc = 100.0 * correct / n
    print(f"\n test: avg loss {avg_loss:.4f}, acc {correct}/{n} ({acc:.1f}%)\n")
    return acc


def get_transforms(dataset_name: str):
    # simple aug + cifar norm (https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
    # https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html
    mean = {
        "cifar10": [0.4914, 0.4822, 0.4465],
        "cifar100": [0.5071, 0.4865, 0.4409],
    }[dataset_name]
    std = {
        "cifar10": [0.2470, 0.2435, 0.2616],
        "cifar100": [0.2673, 0.2564, 0.2762],
    }[dataset_name]

    t_train = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    t_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return t_train, t_test


def main():
    # dataset
    dataset_name = "cifar100"  # cifar100 or the other (cifar10); cifar10 10% random chance, cifar100 1% random chance

    # 6 hour max (learned this rn)
    max_hours = 6.0
    start_time = time.time()
    deadline = start_time + max_hours * 3600

    # training params 
    epochs = 100  #not actually 100 epochs, but goes for how long it can up to the time
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-2

    # device prefer mps on mac
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    # data
    t_train, t_test = get_transforms(dataset_name)
    if dataset_name == "cifar100":
        ds_train = datasets.CIFAR100("./data", train=True, download=True, transform=t_train)
        ds_test = datasets.CIFAR100("./data", train=False, download=True, transform=t_test)
        num_classes = 100
    else:
        ds_train = datasets.CIFAR10("./data", train=True, download=True, transform=t_train)
        ds_test = datasets.CIFAR10("./data", train=False, download=True, transform=t_test)
        num_classes = 10

    # use a decent slice of data to learn more, but not full
    use_subset = True
    subset_frac = 0.5  # about half the train set
    if use_subset:
        n = len(ds_train)
        k = int(n * subset_frac)
        idx = torch.randperm(n)[:k]
        ds_train = torch.utils.data.Subset(ds_train, idx)

    # loaders (no workers on mac)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # model, loss, optim
    model = KANC_CIFAR(grid_size=3, num_classes=num_classes).to(device)
    print(f"model: {model.name}")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, loss_fn, epoch)
        acc = test(model, device, test_loader, loss_fn)
        if acc > best_acc:
            best_acc = acc
            # torch.save(model.state_di

        if time.time() > deadline:
            print("stopping early to respect 6h budget")
            break


if __name__ == "__main__":
    main()
