import numpy as np
import torch

def loader_to_array(loader: torch.utils.data.DataLoader) -> tuple:
    X_parts, y_parts = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            features, labels = batch
        else:
            features, labels = batch, None

        features = features.cpu().numpy()
        if labels is not None:
            labels = labels.cpu().numpy()

        X_parts.append(features)
        if labels is not None:
            y_parts.append(labels)

    X = np.vstack(X_parts) if X_parts else None
    y = np.concatenate(y_parts) if y_parts else None

    return X, y