import numpy as np
import pandas as pd
import torch

from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
from .custom_logger import get_default_logger

logger = get_default_logger()


def moons_dataset(n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Generating moons dataset with {n} samples")
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000, verbose=False):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000, verbose=False):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000, verbose=False):
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x / 54 - 1) * 4
    y = (y / 48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


def california_housing_dataset(verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Loading California Housing dataset")
    california = fetch_california_housing()
    X, y = california.data, california.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    logger.info(f"California Housing dataset loaded. Shape: {X_tensor.shape}")
    return TensorDataset(X_tensor, y_tensor)


def prepare_data(df, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Preparing data from DataFrame")
    data = torch.tensor(df.values, dtype=torch.float32)
    logger.info(f"Data prepared. Shape: {data.shape}")
    return TensorDataset(data)


def get_dataset(name, n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Getting dataset: {name}")
    if name == "moons":
        return moons_dataset(n, verbose)
    elif name == "dino":
        return dino_dataset(n, verbose)
    elif name == "line":
        return line_dataset(n, verbose)
    elif name == "circle":
        return circle_dataset(n, verbose)
    elif name == "california":
        return california_housing_dataset(verbose)
    else:
        logger.error(f"Unknown dataset: {name}")
        raise ValueError(f"Unknown dataset: {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="california",
        choices=["moons", "dino", "line", "circle", "california"],
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, verbose=args.verbose)
    print(f"Dataset {args.dataset} loaded. Shape: {dataset.tensors[0].shape}")
