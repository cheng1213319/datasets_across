import torch.nn as nn
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layers: list,
                 output_size: int = 1,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 use_batchnorm: bool = True,
                 use_layernorm: bool = False,
                 init_method: str = 'he'
                 ):
        super().__init__()
        layers = []
        prev_size = input_size

        for layer_idx, size in enumerate(hidden_layers):
            linear_layer = nn.Linear(prev_size, size)

            if init_method == 'he':
                init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
            elif init_method == 'xavier':
                init.xavier_normal_(linear_layer.weight)

            layers.append(linear_layer)

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(size))
            elif use_layernorm:
                layers.append(nn.LayerNorm(size))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0 and layer_idx != len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))

            prev_size = size

        self.output = nn.Linear(prev_size, output_size)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)


class ModelFactory:
    @staticmethod
    def create(model_type: str, params: Dict[str, Any], input_size: int = None) -> object:
        if model_type == "MLP":
            return ModelFactory._create_mlp(input_size, params)
        elif model_type == "XGB":
            return ModelFactory._create_xgb(params)
        elif model_type == "RF":
            return ModelFactory._create_rf(params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _create_mlp(input_size: int, params: Dict[str, Any]) -> nn.Module:
        required_params = ['hidden_layers', 'lr', 'dropout', 'activation']
        for p in required_params:
            if p not in params:
                raise ValueError(f"MLP requires parameter: {p}")

        model = MLP(
            input_size=input_size,
            hidden_layers=params['hidden_layers'],
            dropout=params.get('dropout', 0.2),
            activation=params.get('activation', 'relu')
        )

        return model

    @staticmethod
    def _create_xgb(params: Dict[str, Any]) -> XGBRegressor:
        base_params = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'learning_rate': params.get('learning_rate', 0.1)
        }

        if params["use_gpu"]:
            base_params.update({
                'tree_method': 'hist',
                'device': 'cuda',
                'n_jobs': 1
            })

        return XGBRegressor(**base_params)

    @staticmethod
    def _create_rf(params: Dict[str, Any]) -> RandomForestRegressor:
        rf_params = params.copy()
        if 'max_depth' in rf_params and rf_params['max_depth'] is None:
            del rf_params['max_depth']

        return RandomForestRegressor(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params.get('max_depth', None),
            **{k: v for k, v in rf_params.items() if k in [
                'min_samples_split', 'max_features', 'bootstrap'
            ]}
        )

    @staticmethod
    def get_model_requirements(model_type: str) -> list:
        requirements = {
            "MLP": ["hidden_layers", "lr"],
            "XGB": ["n_estimators", "max_depth"],
            "RF": ["n_estimators"]
        }
        return requirements.get(model_type, [])

