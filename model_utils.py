import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def calculate_metrics(y_true, y_pred):
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pcc': pearsonr(y_true, y_pred)[0]
    }


def metric(df):
    y_true = df['true']
    y_pred = df['pred']
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pcc": pearsonr(y_true, y_pred)[0]
    }


def calc_stat(numbers):
    mu = sum(numbers) / len(numbers)
    sigma = (sum([(x - mu) ** 2 for x in numbers]) / len(numbers)) ** 0.5
    return mu, sigma


def conf_inv(mu, sigma, n):
    delta = 2.776 * sigma / (n ** 0.5)  # 95%
    return mu - delta, mu + delta

def write_metric_report(prefix, metric_data):
    report = []
    for metric in ['mse', 'mae', 'r2', 'pcc']:
        values = metric_data[metric]
        if not values:
            continue

        mu, sigma = calc_stat(values)
        n = len(values)
        lo, hi = conf_inv(mu, sigma, n)

        if metric == 'r2':
            line = (f"{prefix} R²: {mu:.4f} ± {sigma:.4f} "
                    f"[{lo:.4f}, {hi:.4f}]")
        elif metric == 'pcc':
            line = (f"{prefix} PCC: {mu:.4f} ± {sigma:.4f} "
                    f"[{lo:.4f}, {hi:.4f}]")
        else:
            line = (f"{prefix} {metric.upper()}: {mu:.4f} ± {sigma:.4f} "
                    f"(95% CI: [{lo:.4f}, {hi:.4f}])")

        if metric == 'mse':
            rmse_values = [x ** 0.5 for x in values]
            rmse_mu, rmse_sigma = calc_stat(rmse_values)
            rmse_lo, rmse_hi = conf_inv(rmse_mu, rmse_sigma, n)
            line += (f"\n{prefix} RMSE: {rmse_mu:.4f} ± {rmse_sigma:.4f} "
                     f"(95% CI: [{rmse_lo:.4f}, {rmse_hi:.4f}])")

        report.append(line)

    return "\n".join(report)


def train_sklearn(model, X_train, y_train, X_val, y_val, params):
    model.set_params(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    metrics = calculate_metrics(y_val, preds)
    checkpoint = save_sklearn_model(model, params)
    return metrics['mse'], checkpoint


def train_mlp(model, device, train_loader, optimizer, criterion):
    running_loss = 0.0
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.reshape(labels.shape[0], 1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float()
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        count += 1
        running_loss = 0.0
    return total_loss / count


def valid_mlp(model, device, valid_loader, criterion):
    model.eval()
    compare = pd.DataFrame(columns=('true', 'pred'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(labels.shape[0], 1).to(device)

            outputs = model(inputs)
            labels = labels.float()
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            labels = labels.cpu()
            outputs = outputs.cpu()
            labels_list = np.array(labels)[:, 0].tolist()
            outputs_list = np.array(outputs)[:, 0].tolist()
            compare_temp = pd.DataFrame(columns=('true', 'pred'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = outputs_list
            compare = pd.concat([compare, compare_temp])
    compare_copy = compare.copy()
    metrics = metric(compare_copy)
    return {
        "val_loss": total_loss / count,
        ** metrics
    }


def test_mlp(model, device, test_loader):
    model.eval()
    compare = pd.DataFrame(columns=('true', 'pred'))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(labels.shape[0], 1).to(device)
            outputs = model(inputs)

            labels = labels.cpu()
            outputs = outputs.cpu()
            labels_list = np.array(labels)[:, 0].tolist()
            outputs_list = np.array(outputs)[:, 0].tolist()
            compare_temp = pd.DataFrame(columns=('true', 'pred'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = outputs_list
            compare = pd.concat([compare, compare_temp])
    return compare



def save_mlp_model(epoch, model, optimizer, params):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'params': params
    }

    return checkpoint

def save_sklearn_model(model, params):
    checkpoint = {
        'model': model,
        'params': params
    }

    return checkpoint





