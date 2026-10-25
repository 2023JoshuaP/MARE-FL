from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, Subset


def _as_tensors_from_sklearn_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    x, y = load_digits(return_X_y=True)
    x = x.astype(np.float32) / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    return (
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)),
        64, 10, "mlp",
    )


def _torchvision_dataset(name: str, root: str):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([transforms.ToTensor()])
    root = str(Path(root).expanduser())
    if name == "fashionmnist":
        tr = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        te = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
        return tr, te, 28 * 28, 10, "mlp_image"
    if name == "cifar10":
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
        return tr, te, 32 * 32 * 3, 10, "cnn_cifar"
    raise ValueError(name)


def _uci_har(root: str):
    import io, zipfile, requests
    rootp = Path(root).expanduser() / "uci_har"
    rootp.mkdir(parents=True, exist_ok=True)
    npz = rootp / "har.npz"
    if not npz.exists():
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as outer_z:
            inner_zip_name = None
            for n in outer_z.namelist():
                if n.endswith(".zip"):
                    inner_zip_name = n
                    break
            
            if inner_zip_name:
                z_data = io.BytesIO(outer_z.read(inner_zip_name))
                z = zipfile.ZipFile(z_data)
            else:
                z = outer_z
                
            names = z.namelist()
            def pick(suffix):
                for n in names:
                    if n.endswith(suffix):
                        return n
                raise KeyError(f"No se encontro {suffix} en el ZIP")
            Xtr = np.loadtxt(z.open(pick("train/X_train.txt")), dtype=np.float32)
            ytr = np.loadtxt(z.open(pick("train/y_train.txt")), dtype=np.int64) - 1
            Xte = np.loadtxt(z.open(pick("test/X_test.txt")), dtype=np.float32)
            yte = np.loadtxt(z.open(pick("test/y_test.txt")), dtype=np.int64) - 1
            if inner_zip_name:
                z.close()
                
        np.savez_compressed(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    
    with np.load(npz) as d:
        train = TensorDataset(torch.tensor(d["Xtr"]), torch.tensor(d["ytr"], dtype=torch.long))
        test = TensorDataset(torch.tensor(d["Xte"]), torch.tensor(d["yte"], dtype=torch.long))
        in_dim = int(d["Xtr"].shape[1])
        
    return train, test, in_dim, 6, "mlp"


def load_dataset(name: str, root: str = "~/.cache/marefl"):
    name = name.lower()
    if name == "digits":
        return _as_tensors_from_sklearn_digits()
    if name in {"fashionmnist", "cifar10"}:
        return _torchvision_dataset(name, root)
    if name == "har":
        return _uci_har(root)
    raise ValueError(f"Unknown dataset: {name}")


def extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        y = getattr(dataset, "targets")
        if torch.is_tensor(y):
            return y.cpu().numpy().astype(int)
        return np.asarray(y, dtype=int)
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].cpu().numpy().astype(int)
    return np.asarray([int(y) for _, y in dataset], dtype=int)


def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float, seed: int) -> Tuple[List[Subset], np.ndarray]:
    """Create reproducible non-IID partitions. Histograms are returned only for the
    minimal-repair baseline; MARE-FL does not use entropy or data-value scoring."""
    rng = np.random.default_rng(seed)
    targets = extract_targets(dataset)
    classes = np.unique(targets)
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx, cuts)):
            client_idx[i].extend(part.tolist())
    for i in range(num_clients):
        if not client_idx[i]:
            donor = int(np.argmax([len(v) for v in client_idx]))
            client_idx[i].append(client_idx[donor].pop())
    subsets = [Subset(dataset, sorted(v)) for v in client_idx]
    num_classes = int(targets.max()) + 1
    hists = np.zeros((num_clients, num_classes), dtype=np.float64)
    for i, ids in enumerate(client_idx):
        vals = targets[ids]
        for c in range(num_classes):
            hists[i, c] = np.sum(vals == c)
        if hists[i].sum() > 0:
            hists[i] /= hists[i].sum()
    return subsets, hists
