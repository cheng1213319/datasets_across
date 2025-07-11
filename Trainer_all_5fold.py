import torch
from torch import nn
from model_factory import ModelFactory
from model_utils import train_sklearn, train_mlp, valid_mlp, save_mlp_model
from data_utils import loader_to_array
from datetime import datetime


class Trainer:
    def __init__(self, model_type, device=None):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader, valid_loader, params):
        model = ModelFactory.create(self.model_type, params, input_size=train_loader.dataset[0][0].shape[0])

        if self.model_type == "MLP":
            return self._train_val_mlp(model, train_loader, valid_loader, params)
        else:
            X_train, y_train = loader_to_array(train_loader)
            X_val, y_val = loader_to_array(valid_loader)
            return train_sklearn(model, X_train, y_train, X_val, y_val, params)

    def _train_val_mlp(self, model, train_loader, valid_loader, params):
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=1e-4)

        loss_value = float('inf')
        no_improve_count = 0
        training_logs = []
        checkpoint = None

        for epoch in range(1, params["max_epochs"] + 1):
            train_loss = train_mlp(self.model, self.device, train_loader, self.optimizer, self.criterion)
            val_metrics = valid_mlp(self.model, self.device, valid_loader, self.criterion)
            valid_loss, mse, rmse, mae, r2, pear = val_metrics["val_loss"], val_metrics["mse"], val_metrics["rmse"], val_metrics['mae'], val_metrics["r2"], val_metrics["pcc"]

            if epoch % 5 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                res = (f"[{current_time}] Epoch: {epoch:03d}/{params['max_epochs']}, Train loss: {train_loss:.4f}, Valid loss: {val_metrics['val_loss']:.4f}, "
                       f"MSE: {val_metrics['mse']:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE:{val_metrics['mae']:.4f}, R2: {val_metrics['r2']:.4f}, Pearson: {pear:.4f}")
                print(res)
                training_logs.append(res)

            if valid_loss < loss_value:
                no_improve_count = 0
                loss_value = valid_loss
                checkpoint = save_mlp_model(epoch, self.model, self.optimizer, params)

            else:
                no_improve_count += 1
                if no_improve_count >= params['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return loss_value, checkpoint, training_logs

