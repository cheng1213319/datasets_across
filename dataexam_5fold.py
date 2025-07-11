import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import joblib
from datetime import datetime
from dataset_1.utlis.data_utils import load_data, split
from Trainer_all_5fold import Trainer
import model_utils
import data_utils
from model_factory import ModelFactory
import yaml
from dataset_1.utlis.api import get_drug_combs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def run_fold(repeat, fold, input_path, config, study_name, model_type="MLP", test_mode=True):
    target_name = config['target']
    drug_feat = pd.read_csv(config['drug_feat'])
    cell_feat = pd.read_csv(config['cell_feat'])

    path_train = os.path.join(input_path, f"repeat_{repeat}_fold_{fold}_train.csv")
    path_val = os.path.join(input_path, f"repeat_{repeat}_fold_{fold}_val.csv")

    train_data = load_data(path_train, target_name, drug_feat, cell_feat, fit_preprocessor=True)
    preprocess_params = {
        'drug_scaler': train_data.drug_scaler,
        'cell_scaler': train_data.cell_scaler
    }
    preprocess_path = os.path.join(input_path, f"preprocess_repeat_{repeat}_fold_{fold}_{config['task']}.pkl")
    joblib.dump(preprocess_params, preprocess_path)

    valid_data = load_data(path_val, target_name, drug_feat, cell_feat, ** preprocess_params)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config["batch_size"])

    params_df = pd.read_csv('hyper/results/all_models_performance.csv')
    params_df["params"] = params_df["params"].apply(
        lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x
    )
    params = params_df[params_df["model_type"] == model_type].iloc[0]["params"]
    print(f"Using parameters for {model_type}: {params}")

    trainer = Trainer(model_type, device)
    if model_type == "MLP":
        val_loss, checkpoint, training_logs = trainer.train(train_loader, valid_loader, params)
    else:
        val_loss, checkpoint = trainer.train(train_loader, valid_loader, params)
        training_logs = []

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(
        output_path, str(fold),
        f"checkpoint_{timestamp}.{'pth' if model_type == 'MLP' else 'joblib'}"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if model_type == 'MLP':
        torch.save(checkpoint, checkpoint_path)
    else:
        joblib.dump(checkpoint, checkpoint_path)

    with open(os.path.join(output_path, str(fold), f'repeat{repeat}_fold{fold}.txt'), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {study_name} - {model_type} - {target_name} - repeat{repeat} - fold{fold} --- strat training\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] params: {params}\n")
        for log_entry in training_logs: f.write(log_entry + "\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] End of training\n")
        f.write("-" * 80 + "\n")

    if test_mode:
        test_data = load_data(
            os.path.join(input_path, f"repeat_{repeat}_fold_{fold}_test.csv"),
            target_name,
            drug_feat,
            cell_feat,
        ** joblib.load(preprocess_path)
        )
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        if model_type == "MLP":
            model = ModelFactory.create(model_type, params, test_loader.dataset[0][0].shape[0])
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            compare = model_utils.test_mlp(model, device, test_loader)
        else:
            X_test, y_test = data_utils.loader_to_array(test_loader)
            compare = pd.DataFrame({'true': y_test, 'pred': checkpoint['model'].predict(X_test)})

        compare.to_csv(os.path.join(output_path, str(fold), f"repeat_{repeat}_fold_{fold}_test.csv"), index=False)

        metrics = model_utils.metric(compare)
        with open(os.path.join(output_path, str(fold), f'repeat{repeat}_fold{fold}.txt'), "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test metrics for repeat {repeat}, fold {fold}:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        return checkpoint_path, compare

    return checkpoint_path



if __name__ == '__main__':

    with open('config.yaml', 'r', encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    split_flag = 0  # split data
    # search the best group of hyperparameters

    study_list = [
        'FORCINA', 'MATHEWS', 'NCATS_ES(FAKI/AURKI)', 'NCATS_HL', 'NCATS_DIPG', 'PHELAN', 'BOBROWSKI', 'MOTT',
        'NCATS_SARS-COV-2DPI', 'FRIEDMAN', 'CLOUD', 'FRIEDMAN2', 'ASTRAZENECA', 'FLOBAK','NCATS_ES(NAMPT+PARP)',
        'WILSON', 'ALMANAC', 'YOHE', 'FALLAHI-SICHANI', 'NCATS_ATL', 'DYALL', 'ONEIL', 'MILLER','NCATS_MDR_CS',
        'NCATS_2D_3D', 'SCHMIDT']

    for study in study_list:
        comb_data = get_drug_combs(drug_comb_file='drugcomb_cleaned_with_mean_int.csv', study=study, target_name=config['target'])
        if comb_data.empty:
            print(f"Study {study} has no data in the drugcomb dataset.")
            continue

        comb_data = comb_data[['drug_row', 'drug_col', 'cell_line_name', config['target']]]

        data_dir = os.path.join('data', config['target'], f"{study.replace('/', '_')}")
        os.makedirs(data_dir, exist_ok=True)
        repeat = 1

        if split_flag == 1:
            print('Split data')
            for repeat in range(1, 4):
                split(comb_data, repeat=repeat, input_path=data_dir)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        for model_type in ['MLP', 'XGB', 'RF']:
            output_path = os.path.join('output', model_type, f"{study.replace('/', '_')}_{config['target']}")
            os.makedirs(output_path, exist_ok=True)

            all_metrics = {
                'pcc': [],
                'mse': [],
                'rmse': [],
                'r2': [],
                'mae': []
            }

            for fold in range(1, 6):
                checkpoint_path, _ = run_fold(repeat, fold, data_dir, config, study, model_type)

                with open(os.path.join(output_path, f'repeat{repeat}_all_fold.txt'), "a") as out_f:
                    out_f.write(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {study} - {model_type} - repeat {repeat} - fold {fold}\n")
                    fold_log = os.path.join(output_path, str(fold), f'repeat{repeat}_fold{fold}.txt')
                    with open(fold_log, 'r') as fold_f:
                        lines = [l.strip() for l in fold_f if l.strip()]
                        for line in lines[-5:]:
                            out_f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
                        out_f.write("-"*80 + "\n")


                fold_compare = pd.read_csv(os.path.join(output_path, str(fold), f"repeat_{repeat}_fold_{fold}_test.csv"))
                metrics = model_utils.metric(fold_compare)
                all_metrics['pcc'].append(metrics['pcc'])
                all_metrics['mse'].append(metrics['mse'])
                all_metrics['rmse'].append(metrics['rmse'])
                all_metrics['r2'].append(metrics['r2'])
                all_metrics['mae'].append(metrics['mae'])

            all_report = model_utils.write_metric_report(prefix=None, metric_data=all_metrics)
            with open(os.path.join(output_path, f'repeat{repeat}_all_fold.txt'), "a", encoding='utf-8') as out_f:
                out_f.write("\n=== CROSS-VALIDATION REPORT ===\n")
                out_f.write(f"Study: {study}, Model: {model_type}, Target: {config['target']}\n")
                out_f.write(f"Data numer： {len(comb_data)}\n")
                out_f.write(all_report)
                out_f.write("\n")
                out_f.write("-" * 80 + "\n")









