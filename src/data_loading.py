import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from .custom_logger import get_default_logger

logger = get_default_logger(verbose=False)

def check_and_log_data(X, name):
    logger.info(f"Checking {name} dataset")
    logger.debug(f"{name} shape: {X.shape}")
    logger.debug(f"{name} dtype: {X.dtype}")
    logger.debug(f"{name} min: {np.min(X)}, max: {np.max(X)}")
    logger.debug(f"{name} mean: {np.mean(X)}, std: {np.std(X)}")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    logger.debug(f"{name} NaN count: {nan_count}, Inf count: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"{name} contains NaN or Inf values!")

def moons_dataset(n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Generating moons dataset with {n} samples")
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    check_and_log_data(X, "Moons")
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def line_dataset(n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Generating line dataset with {n} samples")
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    check_and_log_data(X, "Line")
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def circle_dataset(n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Generating circle dataset with {n} samples")
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
    check_and_log_data(X, "Circle")
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def dino_dataset(n=8000, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Generating dino dataset with {n} samples")
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
    check_and_log_data(X, "Dino")
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))

def california_housing_dataset(verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Loading California Housing dataset")
    california = fetch_california_housing()
    X, y = california.data, california.target

    check_and_log_data(X, "California Housing (before scaling)")
    check_and_log_data(y, "California Housing target (before scaling)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    check_and_log_data(X_scaled, "California Housing (after scaling)")

    X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    logger.debug(f"X_tensor stats: min={X_tensor.min().item():.4f}, max={X_tensor.max().item():.4f}, mean={X_tensor.mean().item():.4f}, std={X_tensor.std().item():.4f}")
    logger.debug(f"y_tensor stats: min={y_tensor.min().item():.4f}, max={y_tensor.max().item():.4f}, mean={y_tensor.mean().item():.4f}, std={y_tensor.std().item():.4f}")

    logger.info(f"California Housing dataset loaded. Shape: {X_tensor.shape}")

    return TensorDataset(X_tensor, y_tensor)

def prepare_data(X, y=None, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Preparing data")

    if isinstance(X, pd.DataFrame):
        logger.info("Input is a DataFrame")
        check_and_log_data(X.values, "DataFrame")
        data = torch.tensor(X.values, dtype=torch.float32)
        logger.info(f"Data prepared. Shape: {data.shape}")
        return TensorDataset(data)
    else:
        logger.info("Input is separate X and y")
        check_and_log_data(X, "X")
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if y is not None:
            check_and_log_data(y, "y")
            y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            logger.info(f"Data prepared. X shape: {X_tensor.shape}, y shape: {y_tensor.shape}")
            return TensorDataset(X_tensor, y_tensor)
        else:
            logger.info(f"Data prepared. X shape: {X_tensor.shape}")
            return TensorDataset(X_tensor)

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
