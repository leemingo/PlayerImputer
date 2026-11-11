"""Implements the pass success probability component."""

import itertools
import os
base_path = os.path.abspath(os.path.join(os.getcwd(), "..")) # express/PlayerImputer

from typing import Any, Dict, List, Optional, Union
import math
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from typing import Callable, Dict, List, Optional, Union
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    # root_mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor
from gplearn.genetic import SymbolicClassifier
from sklearn.multioutput import MultiOutputRegressor
from .base import exPressComponent, exPressPytorchComponent
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer                
from .model import SetTransformer, AgentImputer
import PlayerImputer.imputer.config as config

from .calculate import Calculator
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
from math import sqrt, pow
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PressingComponent(exPressComponent):
    """The pressing success probability component."""

    component_name = "pressing"

    def _get_metrics(self, y, y_hat):

        precisions, recalls, thresholds = precision_recall_curve(y, y_hat)

        # F1-score 계산
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        # 최적의 Threshold 찾기
        best_threshold = thresholds[np.argmax(f1_scores)]

        y_pred = (y_hat > best_threshold).astype(int)

        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "log_loss": log_loss(y, y_hat),
            "brier": brier_score_loss(y, y_hat),
            "roc_auc": roc_auc_score(y, y_hat),
        }

class HeatmapComponent(exPressComponent):
    """Base class for an XGBoost-based component."""

    def __init__(self, params):
        super().__init__()
        self.params = params

    def train(self, train_dataloader=None, valid_loader=None) -> Optional[float]:
        return None

    def test(self, dataset) -> Dict[str, float]:
        '''
        전체 평가 지표
        '''
        data_loader = DataLoader(dataset, shuffle=False, batch_size = 1)
        #active_players_dict= dataset.active_players_dict
        active_players_dict_lst = dataset.active_players_dict_lst
        true_labels = []
        pred_labels = []

        preds_lst, targets_lst = [], []
        for _, data in tqdm(enumerate(data_loader)):
            input = data["features"].squeeze(0) # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]
            label = data["labels"].squeeze(0)   # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]
            freeze_frame_mask = data["freeze_frame_mask"].squeeze(0) 

            W, N, Height, Width = input.shape # (1, 22, 68, 105)
            input = input[W//2]
            label = label[W//2]
            freeze_frame_mask = freeze_frame_mask[W//2]

            valid_indices = ~torch.isnan(label).all(dim=1)
            input = input[valid_indices]
            label = label[valid_indices]
            freeze_frame_mask = freeze_frame_mask[valid_indices]

            N, Height, Width = input.shape # (1, 22, 68, 105)
            #print(input.shape, label.shape)
            assert input.shape[0] == label.shape[0] 

            n_away_agents = data["n_away_players_active_lst"]
            actor_valid_index = data["actor_valid_index_lst"]

            # 오차 계산하는 로직
            target_dict = {}
            pred_dict = {}
            active_players_dict = [p for p in active_players_dict_lst[_]]

            for i in range(N):
                # 가중 평균
                row_idx = torch.arange(Height).view(-1, 1).expand(Height, Width)  # shape: (68, 105)
                col_idx = torch.arange(Width).view(1, -1).expand(Height, Width)  # shape: (68, 105)
                avg_row = (row_idx * input[i]).sum()
                avg_col = (col_idx * input[i]).sum()
                
                # 최빈값
                # rows, cols = np.where(input[i] == input[i].max()) # input: 68 x 105
                # avg_row = torch.tensor(rows.mean())
                # avg_col = torch.tensor(cols.mean())  
                x_bin, y_bin = self._get_cell_indexes(avg_col, avg_row)   

                target_dict[active_players_dict[i]] = (label[i][0].item(), label[i][1].item())
                pred_dict[active_players_dict[i]] = (x_bin, y_bin)

            targets_lst.append(target_dict)
            preds_lst.append(pred_dict)

            # 정확도 계산하는 로직
            # Away팀 처리
            cost_matrix = np.zeros((n_away_agents, n_away_agents))
            for i in range(n_away_agents):
                x_bin, y_bin = self._get_cell_indexes(label[i][0], label[i][1])
                cost_matrix[i, :] = input[:n_away_agents, y_bin, x_bin].cpu().numpy()  # away_player x 1

                # weight is probability value, so itself should be maximum weight value 1.
                if (i == actor_valid_index) & (actor_valid_index < n_away_agents):
                    cost_matrix[i, i] = 1 

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)         
            for gt_idx, pred_idx in zip(row_ind, col_ind):
                true_labels.append(gt_idx)    # 실제 선수 ID
                pred_labels.append(pred_idx)  # 매칭된 예측 선수 ID

            # Home팀 처리
            cost_matrix = np.zeros((len(label)-n_away_agents, len(label)-n_away_agents))
            for i in range(n_away_agents, len(label)):
                x_bin, y_bin = self._get_cell_indexes(label[i][0], label[i][1])
                cost_matrix[i-n_away_agents, :] = input[n_away_agents:, y_bin, x_bin].cpu().numpy()  # home_player x 1

                # weight is probability value, so itself should be maximum weight value 1.
                if (i == actor_valid_index) & (actor_valid_index >= n_away_agents):
                    cost_matrix[i-n_away_agents, i-n_away_agents] = 1

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            for gt_idx, pred_idx in zip(row_ind, col_ind):
                true_labels.append(gt_idx+n_away_agents)    # 실제 선수 ID
                pred_labels.append(pred_idx+n_away_agents)  # 매칭된 예측 선수 ID

        a = [t.item()==p.item() for t, p in zip(true_labels, pred_labels)]
        accuracy = sum(a)/len(a)
        print(f"Accuracy: {accuracy*100}")
        
        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (x_pred, y_pred), (x_true, y_true) in zip(preds.values(), targets.values()):
                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
        return accuracy 
                
    def predict(self, loader) -> pd.Series:
        pred_labels = []

        for _, data in enumerate(loader):
            input = data["features"].squeeze(0) # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]
            label = data["labels"].squeeze(0)   # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]

            W, N, _, _ = input.shape # (1, 22, 68, 105)
            input = input[W//2]
            label = label[W//2]
            label = label[~torch.isnan(label).all(dim=1)]

            n_agents = data["n_agents"]
            player_id = data["player_id"]

            # Away팀 처리
            cost_matrix = np.zeros((n_agents, n_agents))
            for i in range(n_agents):
                x_bin, y_bin = self._get_cell_indexes(label[i][0], label[i][1])
                cost_matrix[i, :] = input[:n_agents, y_bin, x_bin].cpu().numpy()  # away_player x 1

                # weight is probability value, so itself should be maximum weight value 1.
                if (i == player_id) & (player_id < n_agents):
                    cost_matrix[i, i] = 1 

            pred_label = []
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)         
            for gt_idx, pred_idx in zip(row_ind, col_ind):
                pred_label.append(pred_idx)  # 매칭된 예측 선수 ID

            # Home팀 처리
            cost_matrix = np.zeros((len(label)-n_agents, len(label)-n_agents))
            for i in range(n_agents, len(label)):
                x_bin, y_bin = self._get_cell_indexes(label[i][0], label[i][1])
                cost_matrix[i-n_agents, :] = input[n_agents:, y_bin, x_bin].cpu().numpy()  # home_player x 1

                # weight is probability value, so itself should be maximum weight value 1.
                if (i == player_id) & (player_id >= n_agents):
                    cost_matrix[i-n_agents, i-n_agents] = 1

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            for gt_idx, pred_idx in zip(row_ind, col_ind):
                pred_label.append(pred_idx+n_agents.item())  # 매칭된 예측 선수 ID
            pred_labels.append(pred_label)

        return pd.DataFrame(pred_labels, columns=range(22))
    
    def predict_freeze_frame(self, dataset) -> Dict[str, float]:
        '''
        freeze_frame 내 선수만 평가 지표
        '''
        data_loader = DataLoader(dataset, shuffle=False, batch_size = 1)
        #active_players_dict= dataset.active_players_dict
        active_players_dict_lst = dataset.active_players_dict_lst
        true_labels = []
        pred_labels = []

        x_errors, y_errors, xy_errors = [], [], []
        in_camera_x_errors, in_camera_y_errors, in_camera_xy_errors = [], [], []

        total_samples = 0
        total_success = 0
        preds_lst, targets_lst = [], []
        for _, data in tqdm(enumerate(data_loader)):
            input = data["features"].squeeze(0) # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]
            label = data["labels"].squeeze(0)   # [bs, seq_len, n_agents, 68, 105] -> [seq_len, n_agents, 68, 105]
            freeze_frame_mask = data["freeze_frame_mask"].squeeze(0) 

            W, N, Height, Width = input.shape # (1, 22, 68, 105)
            input = input[W//2]
            label = label[W//2]
            freeze_frame_mask = freeze_frame_mask[W//2]

            valid_indices = ~torch.isnan(label).all(dim=1)
            input = input[valid_indices]
            label = label[valid_indices]
            freeze_frame_mask = freeze_frame_mask[valid_indices]

            N, Height, Width = input.shape # (1, 22, 68, 105)
            #print(input.shape, label.shape)
            assert input.shape[0] == label.shape[0] 

            n_away_agents = data["n_away_players_active_lst"]
            actor_valid_index = data["actor_valid_index_lst"]

            # 오차 계산하는 로직
            target_dict, pred_dict = {}, {}
            away_pred, home_pred = [], []
            active_players_dict = [p for p in active_players_dict_lst[_]]

            for i in range(N):
                # 가중 평균
                row_idx = torch.arange(Height).view(-1, 1).expand(Height, Width)  # shape: (68, 105)
                col_idx = torch.arange(Width).view(1, -1).expand(Height, Width)  # shape: (68, 105)
                avg_row = (row_idx * input[i]).sum()
                avg_col = (col_idx * input[i]).sum()
                
                # 최빈값
                # rows, cols = np.where(input[i] == input[i].max()) # input: 68 x 105
                # avg_row = torch.tensor(rows.mean())
                # avg_col = torch.tensor(cols.mean())  

                # 평균 값
                x_bin, y_bin = self._get_cell_indexes(avg_col, avg_row)   

                if active_players_dict[i][0] == "A":
                    away_pred.append((x_bin, y_bin))
                elif active_players_dict[i][0] == "H":
                    home_pred.append((x_bin, y_bin))
                else:
                    raise ValueError(f"Not support {active_players_dict[i][0]}")

            # 정확도 계산하는 로직
            # Away팀 처리: 카메라에 포착된 선수만 대상으로
            away_heatmap = input[:n_away_agents]  # [n_valid_pred, H, W]
            away_target = label[:n_away_agents] # [n_valid_target, 2]
            away_freeze_mask = freeze_frame_mask[:n_away_agents]
            assert away_heatmap.shape[0] == away_target.shape[0] == away_freeze_mask.shape[0] == len(away_pred)

            matched_pairs = []
            # actor 제외 (있으면 먼저 1:1 고정 매칭)
            if actor_valid_index < n_away_agents:
                matched_pairs.append((actor_valid_index.item(), actor_valid_index.item()))
                pred_indices = [i for i in range(len(away_heatmap)) if i != actor_valid_index]
                target_indices = [
                    i for i in range(len(away_target)) 
                    if i != actor_valid_index and away_freeze_mask[i]
                ]
            else:
                pred_indices = list(range(len(away_heatmap)))
                target_indices = [
                    i for i in range(len(away_target)) 
                    if away_freeze_mask[i]
                ]

            # freeze된 선수만 대상으로 확률값 기반 매칭
            cost_matrix = np.zeros((len(pred_indices), len(target_indices)))  # (n_pred, n_target)
            for i, pi in enumerate(pred_indices):
                for j, tj in enumerate(target_indices):
                    x_bin, y_bin = self._get_cell_indexes(away_target[tj][0], away_target[tj][1])
                    cost_matrix[i, j] = away_heatmap[pi, y_bin, x_bin].cpu().item()

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            for r, c in zip(row_ind, col_ind):
                real_r = pred_indices[r]
                real_c = target_indices[c]
                matched_pairs.append((real_r, real_c))

            # 매칭 결과 반영
            for r, c in matched_pairs:
                away_pred[r] = away_target[c]

            for r, c in matched_pairs:
                pred_coord = away_pred[c] # 이미 target[c]로 교체된 값
                true_coord = away_target[c]

                in_camera_x_errors.append(abs(pred_coord[0] - true_coord[0]).item())
                in_camera_y_errors.append(abs(pred_coord[1] - true_coord[1]).item())
                in_camera_xy_errors.append((((pred_coord[0] - true_coord[0]) ** 2 + (pred_coord[1] - true_coord[1]) ** 2) ** 0.5).item())

            # Accuracy 계산
            total_samples += len(matched_pairs)
            total_success += len([1 for r, c in matched_pairs if r == c])

            # 홈팀계산
            home_heatmap = input[n_away_agents:]   # (N_pred, H, W)
            home_target = label[n_away_agents:] # (N_target, 2)
            home_freeze_mask = freeze_frame_mask[n_away_agents:]
            assert home_heatmap.shape[0] == home_target.shape[0] == home_freeze_mask.shape[0] == len(home_pred)
            
            matched_pairs = []
            if actor_valid_index >= n_away_agents:
                home_actor_index = (actor_valid_index - n_away_agents).item()
                matched_pairs.append((home_actor_index, home_actor_index))
                pred_indices = [i for i in range(len(home_heatmap)) if i != home_actor_index]
                target_indices = [i for i in range(len(home_target)) if i != home_actor_index and home_freeze_mask[i]]
            else:
                pred_indices = list(range(len(home_heatmap)))
                target_indices = [i for i in range(len(home_target)) if home_freeze_mask[i]]

            cost_matrix = np.zeros((len(pred_indices), len(target_indices)))
            for i, pi in enumerate(pred_indices):
                for j, tj in enumerate(target_indices):
                    x_bin, y_bin = self._get_cell_indexes(home_target[tj][0], home_target[tj][1])
                    cost_matrix[i, j] = home_heatmap[pi, y_bin, x_bin].cpu().item()

            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            for r, c in zip(row_ind, col_ind):
                real_r = pred_indices[r]
                real_c = target_indices[c]
                matched_pairs.append((real_r, real_c))

            for r, c in matched_pairs:
                home_pred[r] = home_target[c]

            for r, c in matched_pairs:
                pred_coord = home_pred[c] # 이미 target[c]로 교체된 값
                true_coord = home_target[c]

                in_camera_x_errors.append(abs(pred_coord[0] - true_coord[0]).item())
                in_camera_y_errors.append(abs(pred_coord[1] - true_coord[1]).item())
                in_camera_xy_errors.append((((pred_coord[0] - true_coord[0]) ** 2 + (pred_coord[1] - true_coord[1]) ** 2) ** 0.5).item())

            # Accuracy 계산
            total_samples += len(matched_pairs)
            total_success += len([1 for r, c in matched_pairs if r == c])

            # 결과 저장
            for idx, (target, pred) in enumerate(zip(away_target, away_pred)): 
                target_dict[active_players_dict[idx]] = (target[0], target[1])
                pred_dict[active_players_dict[idx]] = (pred[0], pred[1])
                
            for idx, (target, pred) in enumerate(zip(home_target, home_pred)):
                target_dict[active_players_dict[idx + n_away_agents]] = (target[0], target[1].item())
                pred_dict[active_players_dict[idx + n_away_agents]] = (pred[0], pred[1])

            targets_lst.append(target_dict)
            preds_lst.append(pred_dict)    
                 
        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (x_pred, y_pred), (x_true, y_true) in zip(preds.values(), targets.values()):
                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")
        print(f"Accuracy: {total_success / total_samples * 100}")

        print("----Error In Camera----")
        print(f"x: {np.mean(in_camera_x_errors)}, y: {np.mean(in_camera_y_errors)}, xy: {np.mean(in_camera_xy_errors)}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
    
class XGBoostComponent(exPressComponent):
    """
    XGBoost-based component for predicting player coordinates.

    The input data is expected to have the shape [batch_size, time_steps, n_agents, num_features],
    and the labels are expected to have the shape [batch_size, n_agents, 2].
    For each player, the features of shape [time_steps, num_features] are flattened into a
    vector of size (time_steps*num_features), and then a MultiOutputRegressor (using XGBRegressor)
    is used to solve the multi-output regression problem (predicting coordinates).
    
    Example params:
        {
            "time_steps": 5,
            "num_features": 15,
            "xgb_params": {
                "objective": "reg:squarederror",
                "n_estimators": 100,
                "random_state": 42
            }
        }
    """
    component_name = "xgboost_regressor"

    def __init__(self, model: XGBRegressor, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.xgb_params = params.get("ModelConfig", {})

        self.models = []
        self.calculator=Calculator()

    def train(self, train_dataset, valid_dataset) -> Optional[float]:
        """
        Trains the XGBoost model using the provided training dataloader.
        Data is extracted from each batch, where each game's player features are flattened
        into a vector and used as input for training.
        """
        cat_idx = train_dataset[0]["categorical_indices"]
                                   
        # Convert the dataset to numpy arrays
        X_train = np.array([x['features'] for x in train_dataset]) # [batch_size, window, n_agents, num_features]

        B, W, N, F = X_train.shape
        y_train = np.array([x['labels'] for x in train_dataset])  # [batch_size, window, n_agents, num_features]
        y_train = y_train[:, W//2, :, :]  # [batch_size, n_agents, label]: # Use only the W//2

        X_train = (
            X_train.transpose(0, 2, 1, 3) # (B, N, W, F)
            .reshape(B*N, W*F) # (B *N, W * F)
        ) 
        y_train = y_train.reshape(B * N, -1)    # [batch_size, n_agents, label]
        
        X_valid = np.array([x['features'] for x in valid_dataset]) # [batch_size, window, n_agents, num_features]
        B, W, N, F = X_valid.shape
        y_valid = np.array([x['labels'] for x in valid_dataset])  # [batch_size, window, n_agents, num_features]
        y_valid = y_valid[:, W//2, :, :]  # [batch_size, n_agents, label]: # Use only the W//2
        X_valid = (
            X_valid.transpose(0, 2, 1, 3) # (B, N, W, F)
            .reshape(B*N, W*F) # (B *N, W * F)
        ) 
        y_valid = y_valid.reshape(B * N, -1)    # [batch_size, n_agents, label]

        # Choose missing value handling strategy based on params
        # Default strategy is "remove", but it can be set to "impute" in the params file.
        missing_strategy = self.xgb_params.get("missing_strategy", "remove")
        print("missing_strategy:",missing_strategy)
        print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
        if missing_strategy == "remove":
            # Option 1: Remove rows with any NaN values.
            X_train_list, y_train_list = [], []
            X_valid_list, y_valid_list = [], []

            mask = ~np.isnan(y_train).any(axis=1)
            X_train = X_train[mask]
            y_train = y_train[mask]

            mask = ~np.isnan(y_valid).any(axis=1)
            X_valid = X_valid[mask]
            y_valid = y_valid[mask]

                # ----- Train -----
            #     X_b = X_train[b]       # [N, W * F]
            #     y_b = y_train[b]       # [N, label]
            #     mask = ~np.isnan(y_b).any(axis=1)
            #     X_b_clean = X_b[mask]
            #     y_b_clean = y_b[mask]

            #     if X_b_clean.shape[0] < 22:
            #         pass
            #         # pad_len = 22 - X_b_clean.shape[0]
            #         # X_pad = np.full((pad_len, X_b_clean.shape[1]), np.nan, dtype=np.float32)
            #         # y_pad = np.full((pad_len, y_b_clean.shape[1]), np.nan, dtype=np.float32)
            #         # X_b_clean = np.vstack([X_b_clean, X_pad])
            #         # y_b_clean = np.vstack([y_b_clean, y_pad])
            #     elif X_b_clean.shape[0] > 22:
            #         raise ValueError("size: ",X_b_clean.shape)
            #     else:
            #         X_train_list.append(X_b_clean.reshape(-1))  # [22 * W * F]
            #         y_train_list.append(y_b_clean.reshape(-1))  # [22 * label]

            # for b in range(X_valid.shape[0]):
            #     # ----- Valid -----
            #     X_bv = X_valid[b]
            #     y_bv = y_valid[b]
            #     mask_v = ~np.isnan(y_bv).any(axis=1)
            #     X_bv_clean = X_bv[mask_v]
            #     y_bv_clean = y_bv[mask_v]

            #     if X_bv_clean.shape[0] < 22:
            #         pass
            #         # pad_len = 22 - X_bv_clean.shape[0]
            #         # X_pad = np.full((pad_len, X_bv_clean.shape[1]), np.nan, dtype=np.float32)
            #         # y_pad = np.full((pad_len, y_bv_clean.shape[1]), np.nan, dtype=np.float32)
            #         # X_bv_clean = np.vstack([X_bv_clean, X_pad])
            #         # y_bv_clean = np.vstack([y_bv_clean, y_pad])
            #     elif X_bv_clean.shape[0] > 22:
            #         raise ValueError("size: ",X_bv_clean.shape)
            #     else:
            #         X_valid_list.append(X_bv_clean.reshape(-1))  # [22 * W * F]
            #         y_valid_list.append(y_bv_clean.reshape(-1))  # [22 * label]

            # X_train = np.stack(X_train_list, axis=0)
            # y_train = np.stack(y_train_list, axis=0)
            # X_valid = np.stack(X_valid_list, axis=0)
            # y_valid = np.stack(y_valid_list, axis=0)
        elif missing_strategy == "impute":
            # Option 2: Impute missing values.
            # Choose imputation strategy: either "zero" or "mean"
            impute_strategy = self.xgb_params.get("impute_strategy", "mean")
            if impute_strategy == "zero":
                # Replace NaNs with zero.
                y_train = np.nan_to_num(y_train, nan=0.0)
            elif impute_strategy == "mean":
                imputer = SimpleImputer(strategy='mean')
                y_train = imputer.fit_transform(y_train)
            else:
                raise ValueError(f"Invalid impute_strategy: {impute_strategy}")
        else:
            raise ValueError(f"Invalid missing_strategy: {missing_strategy}")

        for i in tqdm(range(y_train.shape[1]), desc="training"):
            model = XGBRegressor(**self.xgb_params, eval_metric='rmse')
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_valid, y_valid[:, i])],
                verbose=True
            )
            self.models.append(model)
        return None

    def test(self, test_dataset) -> Dict[str, float]:
        """
        Evaluates the model using the test dataloader.
        The evaluation metric used is Mean Squared Error (MSE).
        """
        self.calculator.total_success = 0
        self.calculator.total_samples = 0
        data_loader = DataLoader(test_dataset, shuffle=False, **self.params["DataConfig"]) # For computing loss
        X_test = np.array([x['features'] for x in test_dataset]) # [batch_size, time_steps, n_agents, num_features]
        B, W, N, F = X_test.shape
        y_test = np.array([x['labels'] for x in test_dataset])  # [batch_size, window, n_agents, num_features]
        y_test = y_test[:, W//2, :, :]  # [batch_size, n_agents, label]: # Use only the W//2
        X_test = (
            X_test.transpose(0, 2, 1, 3) # (B, N, W, F)
            .reshape(B*N, W*F) # (B *N, W * F)
        ) 
        y_test = y_test.reshape(B * N, -1)    # [batch_size, n_agents, label]

        missing_strategy = self.xgb_params.get("missing_strategy", "remove")
        # B, T, N, F = X_test.shape
        # num_outputs = y_test.shape[2]
        # X_test = X_test.reshape(B * N, T * F)
        # y_test = y_test.reshape(B * N, num_outputs)

        if missing_strategy == "remove":
            mask = ~np.isnan(y_test).any(axis=1)
            X_test = X_test[mask]
            y_test = y_test[mask]

            # Option 1: Remove rows with any NaN values.
            # X_test_list, y_test_list = [], []

            # for b in range(B):
            #     X_b = X_test[b]     # [N, F]
            #     y_b = y_test[b]     # [N, output_dim]

            #     mask = ~np.isnan(y_b).any(axis=1)
            #     X_b_clean = X_b[mask]
            #     y_b_clean = y_b[mask]

            #     N_prime = X_b_clean.shape[0]
            #     if N_prime < 22:
            #         pass
            #         # pad_len = 22 - N_prime
            #         # X_pad = np.full((pad_len, X_b_clean.shape[1]), np.nan, dtype=np.float32)
            #         # y_pad = np.full((pad_len, y_b_clean.shape[1]), np.nan, dtype=np.float32)
            #         # X_b_clean = np.vstack([X_b_clean, X_pad])  # [22, F]
            #         # y_b_clean = np.vstack([y_b_clean, y_pad])  # [22, output_dim]
            #     elif X_b_clean.shape[0] > 22:
            #         raise ValueError("size: ",X_b_clean.shape)
            #     else:
            #         X_test_list.append(X_b_clean.reshape(-1))   # [22 * F]
            #         y_test_list.append(y_b_clean.reshape(-1))   # [22 * output_dim]

            # X_test = np.stack(X_test_list, axis=0)  # [B, 22 * F]
            # y_test = np.stack(y_test_list, axis=0)  # [B, 22 * output_dim]

        #y_pred_test = self.model.predict(X_test)
        y_pred_test_x = self.models[0].predict(X_test)  # x 좌표 예측
        y_pred_test_y = self.models[1].predict(X_test)  # y 좌표 예측

        # 스택해서 [N, 2] 형태로 만들기
        y_pred_test = np.stack([y_pred_test_x, y_pred_test_y], axis=1)

        # Calculate Mean Absolute Error (MAE)

        test_mae_x = mean_absolute_error(y_test[:, 0] * config.field_length, y_pred_test[:, 0] * config.field_length)
        test_mae_y = mean_absolute_error(y_test[:, 1] * config.field_width, y_pred_test[:, 1] * config.field_width)
        #test_rmse = root_mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        # Calculate Euclidean Distance Error
        euc_error = np.mean([math.dist(y_pred_test[i], y_test[i]) for i in range(len(y_pred_test))])
        print(f"MAE X: {test_mae_x:.4f}, MAE Y: {test_mae_y:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"Euclidean Distance Error: {euc_error:.4f}")

        # y_test_ori[mask] = y_pred_test
        # y_test_ori = y_test_ori.reshape(B, N, num_outputs)

        # all_preds = torch.Tensor(y_test_ori)  #  리스트 -> Tensor 변환
        # all_targets = torch.cat([batch["labels"] for batch in test_dataset], dim=0).reshape(B, N, num_outputs)  #  targets도 Tensor로 변환
        
        # self.calculator.load_batch_data(data_loader)
        # for i in range(all_preds.shape[0]):      
        #     self.calculator.process_team(all_preds[i], all_targets[i], i)            
        # #  최종 매칭 정확도 계산
        # accuracy = self.calculator.get_accuracy()
        # print(f"Accuracy: {accuracy:.4f}")

        return {
                    "mae_x": test_mae_x,
                    "mae_y": test_mae_y,
                    "rmse": test_rmse,
                    "euclidean_error": euc_error,
                    # "accuracy": accuracy
                }
 
    def predict(self, dataset) -> pd.Series:

        data_loader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"]) # For computing loss
        self.calculator.load_batch_data(data_loader)
        active_players_dict_lst = [batch["active_players_dict_lst"] for batch in data_loader]

        X_test = np.array([x['features'] for x in dataset]) # [batch_size, time_steps, n_agents, num_features]
        B, W, N, F = X_test.shape
        y_test = np.array([x['labels'] for x in dataset])  # [batch_size, window, n_agents, num_features]
        y_test = y_test[:, W//2, :, :]  # [batch_size, n_agents, label]: # Use only the W//2
        X_test = (
            X_test.transpose(0, 2, 1, 3) # (B, N, W, F)
            .reshape(B*N, W*F) # (B *N, W * F)
        ) 
        y_test = y_test.reshape(B * N, -1)    # [batch_size, n_agents, label]

        missing_strategy = self.xgb_params.get("missing_strategy", "remove")

        if missing_strategy == "remove":
            mask = ~np.isnan(y_test).any(axis=1)
            X_test = X_test[mask]
            y_test = y_test[mask]

            # Problem: batch별 각 에이전트의 정보단위가 입력으로 사용되기 때문에 masking이후 batch단위로 복원하는 과정에서 reshape문제가 발생함.
            # 해결: masking과정을 거치기 전 복원할 수 있는 별도의 인덱스를 정리리
            flat_idxs = np.arange(B * N)        # [0, 1, 2, …, B·N−1]
            valid_flat = flat_idxs[mask]        # 예측에 사용된 인덱스들

        preds = np.stack(
            [self.models[0].predict(X_test), self.models[1].predict(X_test)], 
            axis=1
        )
        all_preds = torch.full((B * N, y_test.shape[-1]), float('nan'))
        all_preds[valid_flat] = torch.from_numpy(preds)
        all_preds = all_preds.reshape(B, N, -1)

        all_targets = torch.full((B * N, y_test.shape[-1]), float('nan'))
        all_targets[valid_flat] = torch.from_numpy(y_test)
        all_targets = all_targets.reshape(B, N, -1)
        
        print(all_targets.shape, all_preds.shape)
        for i in range(all_preds.shape[0]):     # all_preds.shape[0]: batch_size
            self.calculator.process_team(all_preds[i], all_targets[i], i)     

        #  최종 매칭 정확도 계산
        accuracy = self.calculator.get_accuracy()
        print("accuracy: ", accuracy)

        preds_lst, targets_lst = [], []
        for i in range(all_preds.shape[0]):      
            preds, targets = all_preds[i], all_targets[i]
            away_targets, away_preds, home_targets, home_preds = self.calculator.process_preds(preds, targets, i)
            
            assert len(away_targets) == len(away_preds), "Away targets and predictions length mismatch"
            assert len(home_targets) == len(home_preds), "Home targets and predictions length mismatch"
            n_away_player = len(away_targets)

            target_dict = {}
            pred_dict = {}

            active_players_dict = [p[0] for p in active_players_dict_lst[i]]
            for idx, (target, pred) in enumerate(zip(away_targets, away_preds)): 
                target_dict[active_players_dict[idx]] = (target[0].item(), target[1].item())
                pred_dict[active_players_dict[idx]] = (pred[0].item(), pred[1].item())
                
            for idx, (target, pred) in enumerate(zip(home_targets, home_preds)):
                target_dict[active_players_dict[idx + n_away_player]] = (target[0].item(), target[1].item())
                pred_dict[active_players_dict[idx + n_away_player]] = (pred[0].item(), pred[1].item())

            
            targets_lst.append(target_dict)
            preds_lst.append(pred_dict)

        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (x_pred, y_pred), (x_true, y_true) in zip(preds.values(), targets.values()):
                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
    
class SetTransformerComponent(exPressPytorchComponent):
    """The pressing success probability component."""

    component_name = "set_transformer"

    def __init__(self, model, params):
        super().__init__(model=model, params=params)

        self.calculator=Calculator()
        
    def test(self, dataset):
        """모델 평가 (mean xy 반환 및 매칭 정확도)"""        
        self.calculator.total_success = 0
        self.calculator.total_samples = 0
        
        data_loader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"])
        self.calculator.load_batch_data(data_loader)
        outputs = self.trainer.predict(self.model, data_loader)#, ckpt_path="best")  # ✅ 예측 수행

        all_preds = torch.cat([o["preds"] for o in outputs], dim=0)
        all_targets = torch.cat([o["labels"] for o in outputs], dim=0)
        actor_global_index = torch.cat([o["actor_global_index"] for o in outputs], dim=0)
        all_visible = torch.cat([o["visible"] for o in outputs], dim=0)

        # error를 측정할 때는 모든 선수 대상으로 계산
        W = all_preds.shape[1]
        #mae, mse_loss, euclidean_dist, cosine_similarity = self.model.criterion(all_preds[:, W//2, :, :].unsqueeze(1), all_targets[:, W//2, :, :].unsqueeze(1), actor_global_index)
        mae, mse_loss, euclidean_dist, cosine_similarity = self.model.criterion(all_preds, all_targets, actor_global_index, all_visible)


        if dataset.use_transform:
            if all_preds.shape[-1] == 4:
                #all_preds[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_preds[:, :2])
                all_preds[..., 2] *= config.field_length
                all_preds[..., 3] *= config.field_width

                #all_targets[:, :2] = dataset.transform.velocity_scaler.inverse_transform(all_targets[:, :2])
                all_targets[..., 2] *= config.field_length
                all_targets[..., 3] *= config.field_width
            elif all_preds.shape[-1] == 2:
                all_preds[..., 0] *= config.field_length
                all_preds[..., 1] *= config.field_width

                all_targets[..., 0] *= config.field_length
                all_targets[..., 1] *= config.field_width
                

        # accuracy를 측정할 때는 포착된 선수만을 대상으로 계산
        # all_targets = torch.where(
        #     freeze_frame_mask[..., None], 
        #     all_targets, 
        #     torch.full_like(all_targets, float('nan'))
        # )

        # evaluation: Total Sequence
        # for i in range(all_preds.shape[0]):     # all_preds.shape[0]: batch_size
        #     for w in range(all_preds.shape[1]): # all_preds.shape[1]: window
        #         self.calculator.process_team(all_preds[i][w], all_targets[i][w], i)   
        
        # evaluation1: T Sequence
        #W = all_preds.shape[1]
        for i in range(all_preds.shape[0]):     # all_preds.shape[0]: batch_size
            self.calculator.process_team(all_preds[i][W//2], all_targets[i][W//2], i)     

        #  최종 매칭 정확도 계산
        accuracy = self.calculator.get_accuracy()

        if len(mae) == 4:
            result = {
                "accuracy": accuracy,
                "vx_error": mae[0].item(),
                "vy_error": mae[1].item(),
                "speed_error": math.sqrt(mae[0].item()**2 + mae[1].item()**2),
                "x_error": mae[2].item(),
                "y_error": mae[3].item(),
                "euc_distance": euclidean_dist.item(),
                "mean_cosine_sim": cosine_similarity.item()   
            }
        # mean_xy_original이 2개일 경우 (x, y만 있
        elif len(mae) == 2:
            result = {
                "accuracy": accuracy,
                "x_error": mae[0].item(),
                "y_error": mae[1].item(),
                "euc_distance": euclidean_dist.item()
            }
        else:
            raise ValueError("Invalid mean_xy shape. Expected 2 or 4 elements: ", len(mae))
        
        return result

    def predict(self, dataset):
        """모델 예측 (x, y 좌표 반환)"""
        """모델 평가 (mean xy 반환 및 매칭 정확도)"""        
        self.calculator.total_success = 0
        self.calculator.total_samples = 0

        self.model.eval()
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        self.calculator.load_batch_data(data_loader)
        active_players_dict_lst = dataset.active_players_dict_lst
        preds_lst, targets_lst = [], []

        with torch.no_grad():
        # `self.trainer.predict()`를 사용하여 모델 예측 수행
            outputs = self.trainer.predict(self.model, dataloaders=data_loader)
            all_preds = torch.cat([o["preds"] for o in outputs], dim=0)    # [batch_size, window, n_agents, 2 or 4]
            all_targets = torch.cat([o["labels"] for o in outputs], dim=0) # [batch_size, window, n_agents, 2 or 4]
            all_actor_global_index = torch.cat([o["actor_global_index"] for o in outputs], dim=0)
            all_visible = torch.cat([o["visible"] for o in outputs], dim=0)
            #freeze_frame_mask = torch.cat([batch["freeze_frame_mask"] for batch in data_loader], dim=0)
            freeze_frame_mask = None
            # all_preds = torch.cat(outputs)  #  리스트 -> Tensor 변환
            # all_targets = torch.cat([batch["labels"] for batch in data_loader], dim=0)  #  targets도 Tensor로 변환

            W = all_preds.shape[1]
            #mae, mse_loss, euclidean_dist, cosine_similarity = self.model.criterion(all_preds[:, W//2, :, :].unsqueeze(1), all_targets[:, W//2, :, :].unsqueeze(1), all_actor_global_index)#, freeze_frame_mask)# freeze_frame_mask[: , W//2, :].unsqueeze(1))
            mae, mse_loss, euclidean_dist, cosine_similarity = self.model.criterion(all_preds, all_targets, all_actor_global_index, all_visible)#, freeze_frame_mask)# freeze_frame_mask[: , W//2, :].unsqueeze(1))
            if len(mae) == 4:
                print(
                    f"vx_error: {mae[0].item()}, "
                    f"vy_error: {mae[1].item()}, \n"
                    f"x_error: {mae[2].item()}, "
                    f"y_error: {mae[3].item()}, \n"
                    f"euc_distance: {euclidean_dist.item()}, "
                    f"mean_cosine_sim: {cosine_similarity.item()}"
                )
            elif len(mae) == 2:
                print(
                    f"x_error: {mae[0].item()}, "
                    f"y_error: {mae[1].item()}, \n"
                    f"euc_distance: {euclidean_dist.item()}"
                )
            else:
                raise ValueError("Invalid mean_xy shape. Expected 2 or 4 elements: ", len(mae))

            if dataset.use_transform:
                if all_preds.shape[-1] == 4:
                    #all_preds[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_preds[:, :2])
                    all_preds[..., 2] *= config.field_length
                    all_preds[..., 3] *= config.field_width

                    #all_targets[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_targets[:, :2])
                    all_targets[..., 2] *= config.field_length
                    all_targets[..., 3] *= config.field_width
                elif all_preds.shape[-1] == 2:
                    all_preds[..., 0] *= config.field_length
                    all_preds[..., 1] *= config.field_width

                    all_targets[..., 0] *= config.field_length
                    all_targets[..., 1] *= config.field_width
                else:
                    raise ValueError("only 2 or 4 shape")
                
            for i in range(all_preds.shape[0]):     # all_preds.shape[0]: batch_size
                self.calculator.process_team(all_preds[i][W//2], all_targets[i][W//2], i)      

            #  최종 매칭 정확도 계산
            accuracy = self.calculator.get_accuracy()
            print(f"Accuracy: {accuracy*100}")

            for i in range(all_preds.shape[0]):      
                preds, targets = all_preds[i][W//2], all_targets[i][W//2]  

                away_targets, away_preds, home_targets, home_preds = self.calculator.process_preds(preds, targets, i)

                assert len(away_targets) == len(away_preds), "Away targets and predictions length mismatch"
                assert len(home_targets) == len(home_preds), "Home targets and predictions length mismatch"
                n_away_player = len(away_targets)

                target_dict = {}
                pred_dict = {}

                active_players_dict = [p for p in active_players_dict_lst[i]]
                for idx, (target, pred) in enumerate(zip(away_targets, away_preds)): 
                    target_dict[active_players_dict[idx]] = (target[0].item(), target[1].item())
                    pred_dict[active_players_dict[idx]] = (pred[0].item(), pred[1].item())
                    
                for idx, (target, pred) in enumerate(zip(home_targets, home_preds)):
                    target_dict[active_players_dict[idx + n_away_player]] = (target[0].item(), target[1].item())
                    pred_dict[active_players_dict[idx + n_away_player]] = (pred[0].item(), pred[1].item())

                targets_lst.append(target_dict)
                preds_lst.append(pred_dict)
                
        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (x_pred, y_pred), (x_true, y_true) in zip(preds.values(), targets.values()):
                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
        # df = pd.DataFrame(all_data, columns=range(22))

        # accuracy = self.calculator.get_accuracy()
        # print(self.calculator.total_samples, self.calculator.total_success)
    
        # return df, accuracy

    def predict_freeze_frame(self, dataset):
        self.model.eval()
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        self.calculator.load_batch_data(data_loader)
        active_players_dict_lst = dataset.active_players_dict_lst #[batch["active_players_dict_lst"] for batch in data_loader]
        freeze_frame_mask_lst = torch.cat([batch["freeze_frame_mask"] for batch in data_loader], dim=0) # (n_events, window, n_agents)
        
        preds_lst, targets_lst = [], []

        total_samples = 0
        total_success = 0

        with torch.no_grad():
        # `self.trainer.predict()`를 사용하여 모델 예측 수행
            outputs = self.trainer.predict(self.model, dataloaders=data_loader)
            all_preds = torch.cat([o["preds"] for o in outputs], dim=0)    # [batch_size, window, n_agents, 2 or 4]
            all_targets = torch.cat([o["labels"] for o in outputs], dim=0) # [batch_size, window, n_agents, 2 or 4]

            all_actor_global_index = torch.cat([o["actor_global_index"] for o in outputs], dim=0)

            all_global_idx = torch.cat([batch["actor_global_index_lst"] for batch in data_loader], dim=0)
            all_local_idx = torch.cat([batch["actor_valid_index_lst"] for batch in data_loader], dim=0)
            all_team_n_away = torch.cat([batch["n_away_players_total_lst"] for batch in data_loader], dim=0)
            all_play_n_away = torch.cat([batch["n_away_players_active_lst"] for batch in data_loader], dim=0)
                    
            if dataset.use_transform:
                if all_preds.shape[-1] == 4:
                    #all_preds[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_preds[:, :2])
                    all_preds[..., 2] *= config.field_length
                    all_preds[..., 3] *= config.field_width

                    #all_targets[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_targets[:, :2])
                    all_targets[..., 2] *= config.field_length
                    all_targets[..., 3] *= config.field_width
                elif all_preds.shape[-1] == 2:
                    all_preds[..., 0] *= config.field_length
                    all_preds[..., 1] *= config.field_width

                    all_targets[..., 0] *= config.field_length
                    all_targets[..., 1] *= config.field_width
                else:
                    raise ValueError("only 2 or 4 shape")

            W = all_preds.shape[1]
            for i in range(all_preds.shape[0]):
                all_away_agents = all_team_n_away[i]
                play_away_agents = all_play_n_away[i]    
                actor_local_index = all_local_idx[i]  
                active_players_dict = [p for p in active_players_dict_lst[i]]
                freeze_frame_mask = freeze_frame_mask_lst[i][W//2]  # [n_agents]

                target_dict, pred_dict = {}, {}

                pred, target = all_preds[i][W//2], all_targets[i][W//2]
                # if pred.shape[-1] == 4:
                #     pred = pred[:, 2:]  # 마지막 두 개 차원(x, y)만 사용
                #     target = target[:, 2:]

                # 원정팀 헝가리안 알고리즘 매칭 -----
                away_pred = pred[:all_away_agents]      # (20,2)
                away_target = target[:all_away_agents]  # (20,2)
                away_freeze_frame = freeze_frame_mask[:all_away_agents]

                valid_indices = ~torch.isnan(away_target).all(dim=1) # target이 NaN인 선수들은 계산에서 제외
                away_pred = away_pred[valid_indices]
                away_target = away_target[valid_indices]
                away_freeze_frame = away_freeze_frame[valid_indices]


                matched_pairs = []
                if actor_local_index < play_away_agents:
                    matched_pairs.append((actor_local_index.item(), actor_local_index.item()))
                    # 1. pred_indices: actor만 제외
                    pred_indices = [i for i in range(away_pred.shape[0]) if i != actor_local_index]

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(away_target.shape[0])
                        if i != actor_local_index and away_freeze_frame[i]
                    ]
                else:
                    # actor가 원정팀에 속하지 않는 경우
                    pred_indices = list(range(away_pred.shape[0]))

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(away_target.shape[0]) 
                        if away_freeze_frame[i]
                    ]


                # 나머지 선수만 헝가리안
                cost = cdist(away_pred[pred_indices][:, -2:], away_target[target_indices][:, -2:], metric='euclidean') # 경기에 참여하는 선수들에 대해서만 계산
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    real_r = pred_indices[r]
                    real_c = target_indices[c]
                    matched_pairs.append((real_r, real_c))

                # 매칭 결과 반영
                for r, c in matched_pairs:
                    away_pred[r][-2:] = away_target[c][-2:]

                # Accuracy 계산
                total_samples += len(matched_pairs)
                total_success += len([1 for r, c in matched_pairs if r == c])
        
                # 홈팀 헝가리안 알고리즘 매칭 -----
                home_pred = pred[all_away_agents:]      # (20,2)
                home_target = target[all_away_agents:]  # (20,2)
                home_freeze_frame = freeze_frame_mask[all_away_agents:]

                valid_indices = ~torch.isnan(home_target).all(dim=1) # target이 NaN인 선수들은 계산에서 제외
                home_pred = home_pred[valid_indices]     # (11, 2)
                home_target = home_target[valid_indices] # (11, 2)
                home_freeze_frame = home_freeze_frame[valid_indices]

                matched_pairs = []
                if actor_local_index >= play_away_agents:
                    home_actor_index = (actor_local_index - play_away_agents).item()
                    matched_pairs.append((home_actor_index, home_actor_index))
                    # 1. pred_indices: actor만 제외
                    pred_indices = [i for i in range(home_pred.shape[0]) if i != home_actor_index]

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(home_target.shape[0])
                        if i != home_actor_index and home_freeze_frame[i]
                    ]
                else:
                    # actor가 원정팀에 속하지 않는 경우
                    pred_indices = list(range(home_pred.shape[0]))

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(home_target.shape[0]) 
                        if home_freeze_frame[i]
                    ]

                cost = cdist(home_pred[pred_indices][:, -2:], home_target[target_indices][:, -2:], metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    real_r = pred_indices[r]
                    real_c = target_indices[c]
                    matched_pairs.append((real_r, real_c))

                for r, c in matched_pairs:
                    home_pred[r][-2:] = home_target[c][-2:]

                # Accuracy 계산
                total_samples += len(matched_pairs)
                total_success += len([1 for r, c in matched_pairs if r == c])

                # 결과 저장
                for idx, (target, pred) in enumerate(zip(away_target, away_pred)): 
                    target_dict[active_players_dict[idx]] = (target[0].item(), target[1].item(), target[2].item(), target[3].item())
                    pred_dict[active_players_dict[idx]] = (pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item())
                    
                for idx, (target, pred) in enumerate(zip(home_target, home_pred)):
                    target_dict[active_players_dict[idx + play_away_agents]] = (target[0].item(), target[1].item(), target[2].item(), target[3].item())
                    pred_dict[active_players_dict[idx + play_away_agents]] = (pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item())

                targets_lst.append(target_dict)
                preds_lst.append(pred_dict)

        vx_errors, vy_errors, vxy_errors = [], [], []
        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (vx_pred, vy_pred, x_pred, y_pred), (vx_true, vy_true, x_true, y_true) in zip(preds.values(), targets.values()):
                if pd.notna(vx_true):
                    vx_errors.append(abs(vx_pred - vx_true))
                    vy_errors.append(abs(vy_pred - vy_true))
                    vxy_errors.append(((vx_pred - vx_true) ** 2 + (vy_pred - vy_true) ** 2) ** 0.5)

                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")
        print(f"vx: {np.mean(vx_errors):.3f}, vy: {np.mean(vy_errors):.3f}, vxy: {np.mean(vxy_errors):.3f}")
        print("Accuracy: ", total_success / total_samples)
        print(f"Accuracy: {total_success / total_samples * 100}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
    
    def predict_freeze_frame1(self, dataset, freeze_frame_mask_lst = None):
        self.model.eval()
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        self.calculator.load_batch_data(data_loader)
        active_players_dict_lst = dataset.active_players_dict_lst #[batch["active_players_dict_lst"] for batch in data_loader]
        
        if freeze_frame_mask_lst is None:
            freeze_frame_mask_lst = torch.cat([batch["freeze_frame_mask"] for batch in data_loader], dim=0) # (n_events, window, n_agents)
        
        preds_lst, targets_lst = [], []
        in_camera_x_errors, in_camera_y_errors, in_camera_xy_errors = [], [], []

        total_samples = 0
        total_success1 = 0
        total_success2 = 0
        
        with torch.no_grad():
        # `self.trainer.predict()`를 사용하여 모델 예측 수행
            outputs = self.trainer.predict(self.model, dataloaders=data_loader)
            all_preds = torch.cat([o["preds"] for o in outputs], dim=0)    # [batch_size, window, n_agents, 2 or 4]
            all_targets = torch.cat([o["labels"] for o in outputs], dim=0) # [batch_size, window, n_agents, 2 or 4]
            all_actor_global_index = torch.cat([o["actor_global_index"] for o in outputs], dim=0)

            all_global_idx = torch.cat([batch["actor_global_index_lst"] for batch in data_loader], dim=0)
            all_local_idx = torch.cat([batch["actor_valid_index_lst"] for batch in data_loader], dim=0)
            all_team_n_away = torch.cat([batch["n_away_players_total_lst"] for batch in data_loader], dim=0)
            all_play_n_away = torch.cat([batch["n_away_players_active_lst"] for batch in data_loader], dim=0)
                    
            if dataset.use_transform:
                if all_preds.shape[-1] == 4:
                    #all_preds[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_preds[:, :2])
                    all_preds[..., 2] *= config.field_length
                    all_preds[..., 3] *= config.field_width

                    #all_targets[..., :2] = dataset.transform.velocity_scaler.inverse_transform(all_targets[:, :2])
                    all_targets[..., 2] *= config.field_length
                    all_targets[..., 3] *= config.field_width
                elif all_preds.shape[-1] == 2:
                    all_preds[..., 0] *= config.field_length
                    all_preds[..., 1] *= config.field_width

                    all_targets[..., 0] *= config.field_length
                    all_targets[..., 1] *= config.field_width
                else:
                    raise ValueError("only 2 or 4 shape")

            W = all_preds.shape[1]
            for i in range(all_preds.shape[0]):
                all_away_agents = all_team_n_away[i]
                play_away_agents = all_play_n_away[i]    
                actor_local_index = all_local_idx[i]  
                active_players_dict = [p for p in active_players_dict_lst[i]]
                freeze_frame_mask = freeze_frame_mask_lst[i][W//2]  # [n_agents]

                target_dict, pred_dict = {}, {}

                pred, target = all_preds[i][W//2], all_targets[i][W//2]
                if pred.shape[-1] == 4:
                    pred = pred[:, 2:]  # 마지막 두 개 차원(x, y)만 사용
                    target = target[:, 2:]

                # 원정팀 헝가리안 알고리즘 매칭 -----
                away_pred = pred[:all_away_agents]      # (20,2)
                away_target = target[:all_away_agents]  # (20,2)
                away_freeze_frame = freeze_frame_mask[:all_away_agents]

                valid_indices = ~torch.isnan(away_target).all(dim=1) # target이 NaN인 선수들은 계산에서 제외
                away_pred = away_pred[valid_indices]
                away_target = away_target[valid_indices]
                away_freeze_frame = away_freeze_frame[valid_indices]


                matched_pairs = []
                if actor_local_index < play_away_agents:
                    matched_pairs.append((actor_local_index.item(), actor_local_index.item()))
                    # 1. pred_indices: actor만 제외
                    pred_indices = [i for i in range(away_pred.shape[0]) if i != actor_local_index]

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(away_target.shape[0])
                        if i != actor_local_index and away_freeze_frame[i]
                    ]
                else:
                    # actor가 원정팀에 속하지 않는 경우
                    pred_indices = list(range(away_pred.shape[0]))

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(away_target.shape[0]) 
                        if away_freeze_frame[i]
                    ]


                # 나머지 선수만 헝가리안
                cost = cdist(away_pred[pred_indices], away_target[target_indices], metric='euclidean') # 경기에 참여하는 선수들에 대해서만 계산
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    real_r = pred_indices[r]
                    real_c = target_indices[c]
                    matched_pairs.append((real_r, real_c))

                # 매칭 결과 반영
                for r, c in matched_pairs:
                    away_pred[r] = away_target[c]

                for r, c in matched_pairs:
                    pred_coord = away_pred[c] # 이미 target[c]로 교체된 값
                    true_coord = away_target[c]

                    in_camera_x_errors.append(abs(pred_coord[0] - true_coord[0]).item())
                    in_camera_y_errors.append(abs(pred_coord[1] - true_coord[1]).item())
                    in_camera_xy_errors.append((((pred_coord[0] - true_coord[0]) ** 2 + (pred_coord[1] - true_coord[1]) ** 2) ** 0.5).item())

                # Accuracy 계산
                total_samples += len(matched_pairs)
                total_success1 += len([1 for r, c in matched_pairs if r == c])
                total_success2 += len([1 for r,c in matched_pairs if away_pred[r].equal(away_target[r])])

        
                # 홈팀 헝가리안 알고리즘 매칭 -----
                home_pred = pred[all_away_agents:]      # (20,2)
                home_target = target[all_away_agents:]  # (20,2)
                home_freeze_frame = freeze_frame_mask[all_away_agents:]

                valid_indices = ~torch.isnan(home_target).all(dim=1) # target이 NaN인 선수들은 계산에서 제외
                home_pred = home_pred[valid_indices]     # (11, 2)
                home_target = home_target[valid_indices] # (11, 2)
                home_freeze_frame = home_freeze_frame[valid_indices]

                matched_pairs = []
                if actor_local_index >= play_away_agents:
                    home_actor_index = (actor_local_index - play_away_agents).item()
                    matched_pairs.append((home_actor_index, home_actor_index))
                    # 1. pred_indices: actor만 제외
                    pred_indices = [i for i in range(home_pred.shape[0]) if i != home_actor_index]

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(home_target.shape[0])
                        if i != home_actor_index and home_freeze_frame[i]
                    ]
                else:
                    # actor가 원정팀에 속하지 않는 경우
                    pred_indices = list(range(home_pred.shape[0]))

                    # 2. target_indices: actor와 freeze_frame=False 모두 제외
                    target_indices = [
                        i for i in range(home_target.shape[0]) 
                        if home_freeze_frame[i]
                    ]

                cost = cdist(home_pred[pred_indices], home_target[target_indices], metric='euclidean')
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    real_r = pred_indices[r]
                    real_c = target_indices[c]
                    matched_pairs.append((real_r, real_c))

                for r, c in matched_pairs:
                    home_pred[r] = home_target[c]

                for r, c in matched_pairs:
                    pred_coord = home_pred[c] # 이미 target[c]로 교체된 값
                    true_coord = home_target[c]

                    in_camera_x_errors.append(abs(pred_coord[0] - true_coord[0]).item())
                    in_camera_y_errors.append(abs(pred_coord[1] - true_coord[1]).item())
                    in_camera_xy_errors.append((((pred_coord[0] - true_coord[0]) ** 2 + (pred_coord[1] - true_coord[1]) ** 2) ** 0.5).item())

                # Accuracy 계산
                total_samples += len(matched_pairs)
                total_success1 += len([1 for r, c in matched_pairs if r == c])
                total_success2 += len([1 for r,c in matched_pairs if home_pred[r].equal(home_target[r])])
           
                # 결과 저장
                for idx, (target, pred) in enumerate(zip(away_target, away_pred)): 
                    target_dict[active_players_dict[idx]] = (target[0].item(), target[1].item())
                    pred_dict[active_players_dict[idx]] = (pred[0].item(), pred[1].item())
                    
                for idx, (target, pred) in enumerate(zip(home_target, home_pred)):
                    target_dict[active_players_dict[idx + play_away_agents]] = (target[0].item(), target[1].item())
                    pred_dict[active_players_dict[idx + play_away_agents]] = (pred[0].item(), pred[1].item())

                targets_lst.append(target_dict)
                preds_lst.append(pred_dict)

        x_errors, y_errors, xy_errors = [], [], []
        for preds, targets in zip(preds_lst, targets_lst):
            for (x_pred, y_pred), (x_true, y_true) in zip(preds.values(), targets.values()):
                x_errors.append(abs(x_pred - x_true))
                y_errors.append(abs(y_pred - y_true))
                xy_errors.append(((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")
        print(f"Accuracy1: {total_success1 / total_samples * 100}")
        print(f"Accuracy2: {total_success2 / total_samples * 100}")

        print("----Error In Camera----")
        print(f"x: {np.mean(in_camera_x_errors)}, y: {np.mean(in_camera_y_errors)}, xy: {np.mean(in_camera_xy_errors)}")

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})

    def predict_freeze_frame2(self, dataset, threshold: float = 1):
        """
        threshold: 거리 임계값(이 값 초과하면 그리디 단계에서 매칭하지 않음)
        """
        self.model.eval()
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        self.calculator.load_batch_data(data_loader)
        active_players_dict_lst = dataset.active_players_dict_lst
        freeze_frame_mask_lst = torch.cat(
            [batch["freeze_frame_mask"] for batch in data_loader], dim=0
        )  # (n_events, window, n_agents)

        preds_lst, targets_lst = [], []

        total_samples = 0
        total_success = 0
        total_match = 0
        total_not_match = 0

        with torch.no_grad():
            outputs = self.trainer.predict(self.model, dataloaders=data_loader)
            all_preds = torch.cat([o["preds"] for o in outputs], dim=0)
            all_targets = torch.cat([o["labels"] for o in outputs], dim=0)
            all_actor_global_index = torch.cat(
                [o["actor_global_index"] for o in outputs], dim=0
            )

            all_global_idx = torch.cat(
                [batch["actor_global_index_lst"] for batch in data_loader], dim=0
            )
            all_local_idx = torch.cat(
                [batch["actor_valid_index_lst"] for batch in data_loader], dim=0
            )
            all_team_n_away = torch.cat(
                [batch["n_away_players_total_lst"] for batch in data_loader], dim=0
            )
            all_play_n_away = torch.cat(
                [batch["n_away_players_active_lst"] for batch in data_loader], dim=0
            )

            if dataset.use_transform:
                if all_preds.shape[-1] == 4:
                    all_preds[..., 2] *= config.field_length
                    all_preds[..., 3] *= config.field_width
                    all_targets[..., 2] *= config.field_length
                    all_targets[..., 3] *= config.field_width
                elif all_preds.shape[-1] == 2:
                    all_preds[..., 0] *= config.field_length
                    all_preds[..., 1] *= config.field_width
                    all_targets[..., 0] *= config.field_length
                    all_targets[..., 1] *= config.field_width
                else:
                    raise ValueError("only 2 or 4 shape")

            W = all_preds.shape[1]
            for i in range(all_preds.shape[0]):
                all_away_agents = all_team_n_away[i].item()
                play_away_agents = all_play_n_away[i].item()
                actor_local_index = all_local_idx[i].item()
                active_players_dict = [p for p in active_players_dict_lst[i]]
                freeze_frame_mask = freeze_frame_mask_lst[i][W // 2]  # [n_agents]

                target_dict, pred_dict = {}, {}

                pred, target = all_preds[i][W // 2], all_targets[i][W // 2]
                if pred.shape[-1] == 4:
                    pred = pred[:, 2:]
                    target = target[:, 2:]

                # ===== 원정팀 매칭 =====
                away_pred = pred[:all_away_agents].clone()  # (N_away, 2)
                away_target = target[:all_away_agents].clone()
                away_freeze_frame = freeze_frame_mask[:all_away_agents]

                valid_mask = ~torch.isnan(away_target).all(dim=1)
                away_pred = away_pred[valid_mask]
                away_target = away_target[valid_mask]
                away_freeze_frame = away_freeze_frame[valid_mask]

                # actor 매칭 처리
                matched_pairs = []
                if actor_local_index < play_away_agents:
                    matched_pairs.append((actor_local_index, actor_local_index))
                    pred_indices = [idx for idx in range(away_pred.shape[0]) if idx != actor_local_index]
                    target_indices = [
                        idx for idx in range(away_target.shape[0])
                        if idx != actor_local_index and away_freeze_frame[idx]
                    ]
                else:
                    pred_indices = list(range(away_pred.shape[0]))
                    target_indices = [
                        idx for idx in range(away_target.shape[0])
                        if away_freeze_frame[idx]
                    ]

                # 그리디 매칭 (threshold 이내인 쌍만)
                sub_preds = away_pred[pred_indices].numpy()
                sub_tgts = away_target[target_indices].numpy()
                dist_matrix = cdist(sub_preds, sub_tgts, metric="euclidean")

                idx_p, idx_t = np.where(dist_matrix < threshold)
                triples = [(p, t, dist_matrix[p, t]) for p, t in zip(idx_p, idx_t)]
                triples.sort(key=lambda x: x[2])

                greedy_matched_pred = set()
                greedy_matched_tgt = set()
                for p_local, t_local, _ in triples:
                    if p_local in greedy_matched_pred or t_local in greedy_matched_tgt:
                        continue
                    greedy_matched_pred.add(p_local)
                    greedy_matched_tgt.add(t_local)
                    real_p = pred_indices[p_local]
                    real_t = target_indices[t_local]
                    matched_pairs.append((real_p, real_t))

                leftover = False
                if leftover:
                    # 남은 인덱스 추출
                    leftover_pred_local = [p for p in range(len(pred_indices)) if p not in greedy_matched_pred]
                    leftover_tgt_local = [t for t in range(len(target_indices)) if t not in greedy_matched_tgt]

                    # 헝가리안 매칭 (남은 것들만)
                    if leftover_pred_local and leftover_tgt_local:
                        rem_preds = away_pred[[pred_indices[p] for p in leftover_pred_local]].numpy()
                        rem_tgts = away_target[[target_indices[t] for t in leftover_tgt_local]].numpy()
                        cost_sub = cdist(rem_preds, rem_tgts, metric="euclidean")
                        row_ind, col_ind = linear_sum_assignment(cost_sub)

                        for r, c in zip(row_ind, col_ind):
                            real_r = pred_indices[leftover_pred_local[r]]
                            real_c = target_indices[leftover_tgt_local[c]]
                            matched_pairs.append((real_r, real_c))

                # 매칭 결과로 pred 좌표를 target 좌표로 대체
                for r, c in matched_pairs:
                    away_pred[r] = away_target[c]

                # 정확도 계산
                total_samples += len(matched_pairs)
                total_success += sum(1 for (r, c) in matched_pairs if r == c)

                #print(target_indices)
                total_match += len([t for t in range(len(target_indices)) if t in greedy_matched_tgt])
                total_not_match += len([t for t in range(len(target_indices)) if t not in greedy_matched_tgt])
                # print("greedy_matched_tgt:",greedy_matched_tgt)
                # print("total_match:",total_match)
                # print("total_not_match:",total_not_match)
                # dd                

                # ===== 홈팀 매칭 =====
                home_pred = pred[all_away_agents:].clone()
                home_target = target[all_away_agents:].clone()
                home_freeze_frame = freeze_frame_mask[all_away_agents:]

                valid_mask = ~torch.isnan(home_target).all(dim=1)
                home_pred = home_pred[valid_mask]
                home_target = home_target[valid_mask]
                home_freeze_frame = home_freeze_frame[valid_mask]

                matched_pairs = []
                if actor_local_index >= play_away_agents:
                    home_actor_index = (actor_local_index - play_away_agents)
                    matched_pairs.append((home_actor_index, home_actor_index))
                    pred_indices = [
                        idx for idx in range(home_pred.shape[0]) if idx != home_actor_index
                    ]
                    target_indices = [
                        idx for idx in range(home_target.shape[0])
                        if idx != home_actor_index and home_freeze_frame[idx]
                    ]
                else:
                    pred_indices = list(range(home_pred.shape[0]))
                    target_indices = [
                        idx for idx in range(home_target.shape[0])
                        if home_freeze_frame[idx]
                    ]

                # 그리디 매칭 (threshold 이내인 쌍만)
                sub_preds = home_pred[pred_indices].numpy()
                sub_tgts = home_target[target_indices].numpy()
                dist_matrix = cdist(sub_preds, sub_tgts, metric="euclidean")
                idx_p, idx_t = np.where(dist_matrix < threshold)
                triples = [(p, t, dist_matrix[p, t]) for p, t in zip(idx_p, idx_t)]
                triples.sort(key=lambda x: x[2])

                greedy_matched_pred = set()
                greedy_matched_tgt = set()
                for p_local, t_local, _ in triples:
                    if p_local in greedy_matched_pred or t_local in greedy_matched_tgt:
                        continue
                    greedy_matched_pred.add(p_local)
                    greedy_matched_tgt.add(t_local)
                    real_p = pred_indices[p_local]
                    real_t = target_indices[t_local]
                    matched_pairs.append((real_p, real_t))

                # 남은 인덱스 추출
                if leftover:
                    leftover_pred_local = [p for p in range(len(pred_indices)) if p not in greedy_matched_pred]
                    leftover_tgt_local = [t for t in range(len(target_indices)) if t not in greedy_matched_tgt]

                    # 헝가리안 매칭 (남은 것들만)
                    if leftover_pred_local and leftover_tgt_local:
                        rem_preds = home_pred[[pred_indices[p] for p in leftover_pred_local]].numpy()
                        rem_tgts = home_target[[target_indices[t] for t in leftover_tgt_local]].numpy()
                        cost_sub = cdist(rem_preds, rem_tgts, metric="euclidean")
                        row_ind, col_ind = linear_sum_assignment(cost_sub)

                        for r, c in zip(row_ind, col_ind):
                            real_r = pred_indices[leftover_pred_local[r]]
                            real_c = target_indices[leftover_tgt_local[c]]
                            matched_pairs.append((real_r, real_c))

                # 매칭 결과로 pred 좌표를 target 좌표로 대체
                for r, c in matched_pairs:
                    home_pred[r] = home_target[c]

                # 정확도 계산
                total_samples += len(matched_pairs)
                total_success += sum(1 for (r, c) in matched_pairs if r == c)

                # 결과 저장 (딕셔너리)
                # away part
                for idx_local, (t_coord, p_coord) in enumerate(zip(away_target, away_pred)):
                    target_dict[active_players_dict[idx_local]] = (
                        t_coord[0].item(),
                        t_coord[1].item(),
                    )
                    pred_dict[active_players_dict[idx_local]] = (
                        p_coord[0].item(),
                        p_coord[1].item(),
                    )

                # home part
                for idx_local, (t_coord, p_coord) in enumerate(zip(home_target, home_pred)):
                    overall_idx = idx_local + play_away_agents
                    target_dict[active_players_dict[overall_idx]] = (
                        t_coord[0].item(),
                        t_coord[1].item(),
                    )
                    pred_dict[active_players_dict[overall_idx]] = (
                        p_coord[0].item(),
                        p_coord[1].item(),
                    )

                preds_lst.append(pred_dict)
                targets_lst.append(target_dict)

        # 최종 오류 및 정확도 출력
        x_errors, y_errors, xy_errors = [], [], []
        for pred_map, tgt_map in zip(preds_lst, targets_lst):
            for (x_p, y_p), (x_t, y_t) in zip(pred_map.values(), tgt_map.values()):
                x_errors.append(abs(x_p - x_t))
                y_errors.append(abs(y_p - y_t))
                xy_errors.append(((x_p - x_t) ** 2 + (y_p - y_t) ** 2) ** 0.5)

        print(f"x: {np.mean(x_errors):.3f}, y: {np.mean(y_errors):.3f}, xy: {np.mean(xy_errors):.3f}")
        print("Success Accuracy: ", total_success / total_samples)

        print("Success Count:", total_match, total_match / (total_match+total_not_match))
        print("Fail Count:", total_not_match, total_not_match / (total_match+total_not_match))

        return pd.DataFrame({"pred": preds_lst, "target": targets_lst})
                             
class AgentImputerLightning(pl.LightningModule):
    def __init__(self, model, scaler=None, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()  # Use Mean Squared Error loss
        self.scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scalers", scaler)
        with open(self.scaler_path, "rb") as f:
            self.label_scaler = pickle.load(f)
        
    def forward(self, x, ts_list, edge_index):
        return self.model(x, ts_list, edge_index)

    def training_step(self, batch, batch_idx):
        input_list, ts_list, targets, global_idx = batch["features"], batch['timestamps'], batch["labels"], batch["player_global_id"]
        ts_list = torch.rand_like(ts_list).to(ts_list.device)
        t1 = [i for i in range(40)]
        edges = [(i, j) for i, j in itertools.product(t1, repeat=2) if i != j]
        src, dst = zip(*edges)  # Unpack list of tuples into two tuples
        edge_index = torch.tensor([src, dst], dtype=torch.long).to(input_list.device)
        
        # Assume batch contains keys: "input_list", "ts_list", "edge_index", "labels"
        # input_list, ts_list, active_players, targets, global_idx = batch["features"], batch['timestamps'], batch['active_players'], batch["labels"], batch["player_global_id"]
        # edges = [(i, j) for i, j in itertools.product(active_players, repeat=2) if i != j]
        # src, dst = zip(*edges)  # Unpack list of tuples into two tuples
        # edge_index = torch.tensor([src, dst], dtype=torch.long).to(input_list.device)
        
        outputs = self.model(input_list, ts_list, edge_index)
        __, rmse_original, mse_loss,_=self.compute_loss(outputs, targets, global_idx)

        mlflow.log_metric("train_mse_loss", mse_loss.item(), step=self.global_step)
        # mlflow.log_metric("train_rmse", rmse_original.item(), step=self.global_step)

        self.log("train_mse_loss", mse_loss, prog_bar=True)
        # self.log("train_rmse", rmse_original, prog_bar=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        input_list, ts_list, targets, global_idx = batch["features"], batch['timestamps'], batch["labels"], batch["player_global_id"]
        ts_list = torch.rand_like(ts_list).to(ts_list.device)
        t1 = [i for i in range(40)]
        edges = [(i, j) for i, j in itertools.product(t1, repeat=2) if i != j]
        src, dst = zip(*edges)  # Unpack list of tuples into two tuples
        edge_index = torch.tensor([src, dst], dtype=torch.long).to(input_list.device)
        # input_list, ts_list, active_players, targets, global_idx = batch["features"], batch['timestamps'], batch['active_players'], batch["labels"], batch["player_global_id"]
        # edges = [(i, j) for i, j in itertools.product(active_players, repeat=2) if i != j]
        # src, dst = zip(*edges)  # Unpack list of tuples into two tuples
        # edge_index = torch.tensor([src, dst], dtype=torch.long).to(input_list.device)
        
        outputs = self.model(input_list, ts_list, edge_index)
        __, rmse_original, mse_loss, _=self.compute_loss(outputs, targets, global_idx)
        mlflow.log_metric("valid_mse_loss", mse_loss.item(), step=self.global_step)
        #mlflow.log_metric("valid_rmse", rmse_original.item(), step=self.global_step)

        # self.log("valid original rmse", rmse_original, prog_bar=True)
        self.log("valid_mse_loss", mse_loss, prog_bar=True)
        return mse_loss

    def predict_step(self, batch, batch_idx):
        input_list, ts_list = batch["features"], batch["timestamps"]
        ts_list = torch.rand_like(ts_list).to(ts_list.device)
        t1 = [i for i in range(40)]
        edges = [(i, j) for i, j in itertools.product(t1, repeat=2) if i != j]
        src, dst = zip(*edges)  # Unpack list of tuples into two tuples
        edge_index = torch.tensor([src, dst], dtype=torch.long).to(input_list.device)
        return self.model(input_list, ts_list, edge_index)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def compute_loss(self, preds, targets, global_idx):
        """
        MSE Loss를 계산하면서, global_idx에 해당하는 데이터를 제외함.
        """
        feat_dim = targets.shape[-1]
        mask = ~torch.isnan(targets)
        batch_indices = torch.arange(targets.shape[0], device=targets.device)
        mask[batch_indices, global_idx]= False 
        preds_masked = preds[mask].view(-1, feat_dim)
        targets_masked = targets[mask].view(-1, feat_dim)
    
        mse_loss = F.mse_loss(preds_masked, targets_masked)  # mse
        rmse = torch.sqrt(torch.clamp(mse_loss, min=1e-8))  # rmse

        xy_diff = torch.abs(preds_masked - targets_masked)
        mean_xy =xy_diff.mean(dim=0)
        
        euc_error = torch.norm(preds_masked[:, 2:] - targets_masked[:, 2:], dim=1)  # (x
        mean_euc_error = euc_error.mean()  # 평균 Euclidean Distance 계산

        return mean_xy, rmse, mse_loss, mean_euc_error
    
class AgentImputerComponent(exPressPytorchComponent):
    """The pressing success probability component."""

    component_name = "agent_imputer"

    def __init__(self, params, dataset, scaler_path):
        sample=dataset[0]
        sample_input, sample_output, sample_ts, categorical_indices, global_idx= sample["features"], sample["labels"], sample['timestamps'], sample["categorical_indices"], sample["player_global_id"]
       
        dim_input = sample_input.shape[-1]  # 입력 차원 (Feature 개수)
        dim_output = sample_output.shape[-1]  # 출력 차원 (x, y 좌표 예측)
        num_players = 40  # 축구 경기 선수 수

        dim_hidden = params.get("dim_hidden", 128)

        model = AgentImputer(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            output_size=2,
            categorical_indices=categorical_indices
        )
        lightning_model = AgentImputerLightning(model, lr=params.get("lr", 1e-3), scaler="labels_scaler.pkl")
        self.calculator=Calculator()
        velocity_scaler_path = os.path.join(scaler_path, "velocity_label_scaler.pkl")
        with open(velocity_scaler_path, "rb") as f:
            self.velocity_scaler = pickle.load(f)

        super().__init__(model=lightning_model, params=params)

    def train(self, train_dataset, valid_dataset):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()
        
        train_loader= DataLoader(train_dataset, shuffle=True, **self.params["DataConfig"])
        valid_loader = DataLoader(valid_dataset, shuffle=False, **self.params["DataConfig"])

        checkpoint_callback = ModelCheckpoint(
            monitor="valid_rmse",
            dirpath=self.save_path,
            filename="best_model",
            save_top_k=1,
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="valid_rmse",
            patience=10,
            mode="min",
        )

        self.trainer = pl.Trainer(
            max_epochs=self.params.get("epochs", 30),
            callbacks=[checkpoint_callback, early_stop_callback],
             enable_progress_bar=True,  # ✅ 학습 진행 표시
            logger=False,  # ✅ Lightning 기본 로거 대신 MLflow 사용
            **self.params["TrainerConfig"]
        )
        run = mlflow.active_run()
        if run is None:
            mlflow.start_run()
            
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )

        mlflow.end_run()

    def test(self, dataset):
        """모델 평가 (mean xy 반환 및 매칭 정확도)"""
        self.calculator.total_success = 0
        self.calculator.total_samples = 0
        data_loader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"])
        outputs = self.trainer.predict(self.model, dataloaders=data_loader)
        all_preds = torch.cat(outputs)  #  리스트 -> Tensor 변환
        all_targets = torch.cat([batch["labels"] for batch in data_loader], dim=0)  #  targets도 Tensor로 변환
        all_actor_global_index = torch.cat([batch["actor_global_index"] for batch in data_loader], dim=0)  #  global_idx 추가


        n_away_player=torch.cat([batch["n_away_players"] for batch in data_loader], dim=0)  # n_away_player 추가
        mean_xy_original, rmse_original, __, mean_euc_error = self.model.compute_loss(all_preds, all_targets, actor_global_index)

        self.calculator.load_batch_data(data_loader)

        for i in range(all_preds.shape[0]):      
            self.calculator.process_team(all_preds[i], all_targets[i], i)            
        #  최종 매칭 정확도 계산
        accuracy = self.calculator.get_accuracy()
        scaling_value = sqrt(pow(105, 2) + pow(68, 2))
        
        
        metrics = {
                "accuracy": accuracy,
                "rmse": rmse_original.item(), 
                "euc_distance": mean_euc_error.item() * scaling_value
            }
        if all_targets.shape[-1] == 2:
            metrics.update({
                "x_error": mean_xy_original[0].item() * 105,
                "y_error": mean_xy_original[1].item() * 68,
            })
        elif all_targets.shape[-1] == 4:
            
            normalized_velocity=mean_xy_original[:2].unsqueeze(0)
            original_velocity = self.velocity_scaler.inverse_transform(normalized_velocity)
            metrics.update({
                "vx_error": original_velocity[0][0],
                "vy_error": original_velocity[0][1],
                "x_error": mean_xy_original[2].item() * 105,
                "y_error": mean_xy_original[3].item() * 68,
            })
        return metrics

    def predict(self, dataset):
        """모델 예측 (x, y 좌표 반환)"""
        data_loader = DataLoader(dataset, shuffle=False, **self.params["DataConfig"])

        # `self.trainer.predict()`를 사용하여 모델 예측 수행
        outputs = self.trainer.predict(self.model, dataloaders=data_loader)

        # 모든 배치의 출력을 하나의 배열로 변환
        all_preds = torch.cat(outputs,dim=0)
        return all_preds
    

    