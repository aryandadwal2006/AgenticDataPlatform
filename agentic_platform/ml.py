from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def train_models(path_value: str, task: str, target: str, clusters: int = 3) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    data = pd.read_csv(Path(path_value).expanduser())
    if task != "clustering" and target not in data.columns:
        raise ValueError(f"target column not found: {target}")
    if task == "clustering":
        if len(data) <= clusters:
            raise ValueError("clustering requires more rows than clusters")
        x = pd.get_dummies(data.drop(columns=[target], errors="ignore"), dummy_na=True).fillna(0).to_numpy(dtype=np.float32)
        scaled = StandardScaler().fit_transform(x)
        model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        silhouette = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 and len(labels) > clusters else None
        return {"task": task, "clusters": clusters, "inertia": float(model.inertia_), "silhouette": silhouette}

    x_frame = pd.get_dummies(data.drop(columns=[target]), dummy_na=True).fillna(0)
    x = x_frame.to_numpy(dtype=np.float32)
    if task == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(data[target].astype(str)).astype(np.int64)
        output_size = len(encoder.classes_)
        if output_size < 2:
            raise ValueError("classification requires at least two target classes")
    elif task == "regression":
        y = pd.to_numeric(data[target], errors="raise").to_numpy(dtype=np.float32)
        output_size = 1
    else:
        raise ValueError("task must be classification, regression, or clustering")
    class_counts = np.bincount(y) if task == "classification" else np.array([])
    if len(y) < 5:
        raise ValueError("at least five rows are required for a held-out evaluation")
    stratify = y if task == "classification" and len(class_counts) > 1 and class_counts.min() >= 2 and len(class_counts) <= int(len(y) * 0.2) else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=stratify
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    results: dict[str, Any] = {"task": task, "features": int(x.shape[1]), "train_rows": int(len(x_train)), "test_rows": int(len(x_test))}

    import tensorflow as tf
    tf.random.set_seed(42)
    activation = "softmax" if task == "classification" and output_size > 2 else ("sigmoid" if task == "classification" else "linear")
    tf_units = 1 if task == "classification" and output_size == 2 else (output_size if task == "classification" else 1)
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(tf_units, activation=activation),
    ])
    if task == "classification":
        loss = "sparse_categorical_crossentropy" if output_size > 2 else "binary_crossentropy"
        y_tf_train = y_train.reshape(-1, 1).astype(np.float32) if output_size == 2 else y_train.astype(np.int64)
    else:
        loss = "mse"
        y_tf_train = y_train.reshape(-1, 1)
    tf_model.compile(optimizer="adam", loss=loss)
    tf_model.fit(x_train, y_tf_train, epochs=10, batch_size=32, verbose=0)
    tf_pred = tf_model.predict(x_test, verbose=0)
    if task == "classification":
        tf_labels = (tf_pred.reshape(-1) >= 0.5).astype(int) if output_size == 2 else tf_pred.argmax(axis=1)
        results["tensorflow"] = {"accuracy": float(accuracy_score(y_test, tf_labels)), "f1": float(f1_score(y_test, tf_labels, average="weighted", zero_division=0))}
    else:
        tf_pred = tf_pred.reshape(-1)
        results["tensorflow"] = {"rmse": float(np.sqrt(mean_squared_error(y_test, tf_pred))), "mae": float(mean_absolute_error(y_test, tf_pred)), "r2": float(r2_score(y_test, tf_pred))}

    import torch
    from torch import nn
    torch.manual_seed(42)
    class Network(nn.Module):
        def __init__(self, input_size: int, output_size: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, output_size))
        def forward(self, values: Any) -> Any:
            return self.layers(values)
    torch_model = Network(x_train.shape[1], output_size if task == "classification" else 1)
    xtr = torch.tensor(x_train, dtype=torch.float32)
    xte = torch.tensor(x_test, dtype=torch.float32)
    if task == "classification":
        ytr = torch.tensor(y_train, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
    else:
        ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
    for _ in range(10):
        torch_model.train()
        for start in range(0, len(xtr), 32):
            optimizer.zero_grad()
            prediction = torch_model(xtr[start:start + 32])
            loss_value = criterion(prediction, ytr[start:start + 32])
            loss_value.backward()
            optimizer.step()
    torch_model.eval()
    with torch.no_grad():
        torch_pred = torch_model(xte).cpu().numpy()
    if task == "classification":
        torch_labels = torch_pred.argmax(axis=1)
        results["pytorch"] = {"accuracy": float(accuracy_score(y_test, torch_labels)), "f1": float(f1_score(y_test, torch_labels, average="weighted", zero_division=0))}
    else:
        torch_pred = torch_pred.reshape(-1)
        results["pytorch"] = {"rmse": float(np.sqrt(mean_squared_error(y_test, torch_pred))), "mae": float(mean_absolute_error(y_test, torch_pred)), "r2": float(r2_score(y_test, torch_pred))}
    return results


