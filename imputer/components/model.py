import os
import sys
base_path = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(base_path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datatools.preprocess import extract_match_id
from imputer.datasets import ImputerDataset
from torch.utils.data import DataLoader, RandomSampler
import torch
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import pytorch_lightning as pl
from typing import Any, Dict, List, Optional, Union
import copy
import math
from xgboost import XGBRegressor
import imputer.config as config  # categorical feature 목록 가져오기
import mlflow
mlflow.pytorch.autolog()

from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class PositionalEncoding(nn.Module):
#     """Positional encoding.

#     Args:
#         d_model: Embedding dimension.
#         dropout_rate: Dropout rate.
#         max_len: Maximum input length.
#         reverse: Whether to reverse the input position.
#     """

#     def __init__(self, d_model, dropout_rate=0.1, max_len=5000, reverse=False):
#         """Construct an PositionalEncoding object."""
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.reverse = reverse
#         self.xscale = math.sqrt(self.d_model)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.pe = None
#         self.extend_pe(torch.tensor(0.0).expand(1, max_len))

#     def extend_pe(self, x):
#         """Reset the positional encodings."""
#         if self.pe is not None:
#             if self.pe.size(1) >= x.size(1):
#                 if self.pe.dtype != x.dtype or self.pe.device != x.device:
#                     self.pe = self.pe.to(dtype=x.dtype, device=x.device)
#                 return
#         pe = torch.zeros(x.size(1), self.d_model)
#         if self.reverse:
#             position = torch.arange(
#                 x.size(1) - 1, -1, -1.0, dtype=torch.float32
#             ).unsqueeze(1)
#         else:
#             position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, self.d_model, 2, dtype=torch.float32)
#             * -(math.log(10000.0) / self.d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.pe = pe.to(device=x.device, dtype=x.dtype)

#     def forward(self, x: torch.Tensor):
#         """Add positional encoding.
#         Args:
#             x (torch.Tensor): Input tensor B X T X C
#         Returns:
#             torch.Tensor: Encoded tensor B X T X C
#         """
#         self.extend_pe(x)
#         x = x * self.xscale + self.pe[:, : x.size(1)]
#         return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O=self.fc_o(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    
class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        # FFN 부분
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 정규화 및 dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, query, key, value, src_mask=None, src_key_padding_mask=None):
        # Self-attention (혹은 cross-attention처럼 사용 가능)
        attn_output, _ = self.self_attn(query, key, value, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = query + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])  # 깊이마다 독립된 layer
        self.norm = nn.LayerNorm(encoder_layer.linear2.out_features)

    def forward(self, query, key, value, src_key_padding_mask=None):
        output = query
        for layer in self.layers:
            output = layer(output, key, value, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)
    
# class SAB(nn.Module):
#     def __init__(self, dim_in, dim_out, num_heads, dim_feedforward, ln=False):
#         super(SAB, self).__init__()
#         self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim_out, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, dim_out)
#         )
#         self.ln = nn.LayerNorm(dim_out) if ln else nn.Identity()

    # def forward(self, X):
    #     X=self.mab(X, X)
    #     X = self.ln(X + self.ffn(X))  # Residual + LayerNorm
    #     return X
    # def forward(self, X):
    #     H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
    #     return self.mab1(X, H)
    
class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        X=self.mab(X, X)
        return X


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)    

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)  

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
class AgentAwareAttention(nn.Module): # 
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, D)
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x
    
class AgentAwareBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim=512):
        super().__init__()
        self.attn = AgentAwareAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.norm(x + self.ffn(x))  # Residual connection
        return x
    
class SetTransformer(nn.Module):
    """Identity-based set transformer.

    Every player has one token in every snapshot (identity is given, so there is no association step).
    A token carries the player's feature vector X (role/side/... embeddings) plus the snapshot observation:
    obs_xy (x, y) and a visible flag. Hidden players (visible=0) get obs_xy zeroed, so a hidden token cannot read
    a stale coordinate; it infers its state through self-attention over the identified teammates and opponents.
    Positions of visible players are predicted as obs_xy + delta.

    Inputs (x, y in the same normalised units as the position labels, i.e. divided by field length / width):
        X:       (B, W, N, F) per-player features
        obs_xy:  (B, W, N, 2) observed (x, y); values at hidden players are ignored (may be NaN)
        visible: (B, W, N)    1/True if the player is in the snapshot
        valid:   (B, W, N)    optional; True where the player exists at that timestep, False for padding slots / absent timesteps
                              (excluded as attention key in both the player axis and the time axis)
    Output: (B, W, N, dim_output); dim_output 6 -> (vx, vy, x, y, ax, ay), 4 -> (vx, vy, x, y), 2 -> (x, y)
    """
    def __init__(self, dim_input, num_outputs, dim_output, num_inds=32, 
                 dim_hidden=256, num_heads=8, dim_feedforward=1024, dropout=0.1, 
                 num_seeds=4, num_layers=6, ln=False, categorical_indices=None, categorical_features=None):
        
        super(SetTransformer, self).__init__()

        # {이름: 클래스 수}. None이면 config.categorical_features. categorical_indices의 열 순서는 이름을 정렬한 순서와 같아야 함
        self.categorical_features = categorical_features if categorical_features is not None else config.categorical_features

        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.num_indices = [i for i in range(dim_input) if i not in self.categorical_indices]

        self.dim_output = dim_output

        self.categorical_embeddings = nn.ModuleDict()
        for cat in self.categorical_features:
            num_classes = self.categorical_features[cat]  # 클래스 개수
            embed_dim = int(math.sqrt(num_classes))  # 임베딩 차원 설정
            self.categorical_embeddings[cat] = nn.Embedding(num_classes+1, embed_dim, padding_idx=0)  # 0은 패딩 인덱스

        # 기존 feature와 결합할 임베딩 차원 계산
        total_embedding_dim = sum([emb.embedding_dim for emb in self.categorical_embeddings.values()])
        new_input_dim = dim_input - len(self.categorical_indices) + total_embedding_dim  # 수정

        self.input_fc = nn.Linear(new_input_dim + 3, dim_hidden)  # +3: obs_x, obs_y, visible

        # batch_first=False: (Seq, Batch, Feature) -> Spatial Transformer관점에서는 (N, B*W, F)
        self.player_encoder1 = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 8, dropout, activation="gelu"),
            num_layers,
        )

        # batch_first=False: (Seq, Batch, Feature) -> Temporal Transformer관점에서는 (W, B*N, F)
        self.time_pos_encoder = PositionalEncoding(dim_hidden)
        self.time_encoder1 = TransformerEncoder(
            TransformerEncoderLayer(dim_hidden, num_heads, dim_hidden * 8, dropout, activation="gelu"),
            num_layers,
        )

        self.fc_v = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU(), nn.Linear(dim_hidden, 2))  # 속도 예측용

        self.fc_xy = nn.Sequential(  # 위치 예측: 보이는 선수는 obs_xy에 더해지는 residual, 가려진 선수는 절대 좌표
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, 2),
        ) 

        if dim_output == 6:
            self.fc_a = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.GELU(), nn.Linear(dim_hidden, 2))  # 가속도 예측용

    def forward(self, X, obs_xy, visible, valid=None):
        B, W, N, F = X.shape  # (Batch, Window, Players, Features)

        # 가려진 선수의 관측 좌표는 0으로 만든다 (NaN이어도 where가 걸러냄). 이 값이 residual의 기준점이 되므로 가려진 선수에는 residual이 더해지지 않는다.
        vis = visible.bool().unsqueeze(-1)  # (B, W, N, 1)
        obs_xy = torch.where(vis, obs_xy, torch.zeros_like(obs_xy))  # (B, W, N, 2)

        # 연속형 feature와 카테고리 feature 분리
        X_continuous = X[..., self.num_indices]  # 연속형 변수
        X_categorical = X[..., self.categorical_indices].long()  

        # 카테고리컬 변수 임베딩 적용
        embedded_features = []

        if len(self.categorical_indices) != 0:
            for i, cat in enumerate(sorted(self.categorical_features.keys())):  # 순서 보장
                embedded_features.append(self.categorical_embeddings[cat](X_categorical[..., i]))
            
        # 임베딩된 feature들을 결합
        if embedded_features:
            X_embedded = torch.cat(embedded_features, dim=-1)
            X = torch.cat([X_continuous, X_embedded], dim=-1)

        # Step1. input projection: 선수 token = [identity/role feature, obs_xy, visible]
        X = self.input_fc(torch.cat([X, obs_xy, vis.to(obs_xy.dtype)], dim=-1))  # (B, W, N, dim_hidden)

        # valid: padding slot은 attention에서 제외. 전부 padding인 스냅샷은 NaN 방지를 위해 마스크하지 않음
        pad, pad_t = None, None
        if valid is not None:
            pad = ~valid.bool()
            pad_t = pad.permute(0, 2, 1).reshape(B * N, W)  # 시간축: 선수별로 존재하지 않는 timestep을 key에서 제외 (B*N, W)
            pad_t = pad_t & ~pad_t.all(dim=-1, keepdim=True)
            pad = pad & ~pad.all(dim=-1, keepdim=True)
            pad = pad.reshape(B * W, N)  # (B*W, N), True인 위치는 attention에서 제외

        for i in range(4):
            # Step 2. spatial transformer: 모든 선수 token 사이의 self-attention
            X = (
                X.permute(2, 0, 1, 3)  # (B, W, N, dim_hidden) -> (N, B, W, dim_hidden)
                .contiguous()  # 메모리 연속성 확보
                .view(N, B * W, -1)  # (N, B, W, dim_hidden) -> (N, B*W, dim_hidden)
            )
            X = self.player_encoder1(X, src_key_padding_mask=pad)  # (3) self-attention(Q=K=V=player tokens)

            # Step 3. temporal transformer
            X = (
                X.view(N, B, W, -1)  # (N, B*W, dim_hidden) -> (N, B, W, dim_hidden)
                .permute(2, 1, 0, 3)  # (N, B, W, dim_hidden) -> (W, B, N, dim_hidden)
                .contiguous()  # 메모리 연속성 확보
                .view(W, B * N, -1)  # (W, B, N, dim_hidden) -> (W, B*N, dim_hidden)
            )
        
            X = self.time_pos_encoder(X) 
            X = self.time_encoder1(X, src_key_padding_mask=pad_t)  # (7) self-attention: (W, B*N, dim_hidden)
   
            X = (
                X.view(W, B, N, -1)  # (W, B*N, dim_hidden) -> (W, B, N, dim_hidden)
                .permute(1, 0, 2, 3)  # (W, B, N, dim_hidden) -> (B, W, N, dim_hidden)
                .contiguous()  # 메모리 연속성 확보
            )

        xy_pred = self.fc_xy(X) + obs_xy  # 보이는 선수: obs_xy + delta, 가려진 선수: obs_xy=0이므로 delta (절대 좌표)
        if self.dim_output == 4:
            output = torch.cat([self.fc_v(X), xy_pred], dim=-1)  # (B, W, N, 4)
        elif self.dim_output == 6:
            output = torch.cat([self.fc_v(X), xy_pred, self.fc_a(X)], dim=-1)  # (B, W, N, 6): (vx, vy, x, y, ax, ay)
        else:
            output = xy_pred  # (B, W, N, 2)  ->  (x,y)

        return output

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=True, bidirectional=True):
        # assumes that batch_first is always true
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs

class GCN(torch.nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv1 = SAGEConv(input_size, 64)
        self.conv2 = SAGEConv(64, 32)
        self.linear2 = nn.Linear(32,2) 
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

class seq_lstm(nn.Module):
    def __init__(self, input_size=66, hidden_layer_size=100, output_size=50, batch_size=128):
        super().__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.lstm = TimeLSTM(input_size, hidden_layer_size, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.batch_size = batch_size
        self.relu = nn.ReLU()

    def forward(self, input_seq, ts):
        lstm_out = self.lstm(input_seq,ts)
        outs = self.linear(lstm_out[:,-1,:])
        outs = self.relu(outs)
        return outs

# --- loss 설정 -------------------------------------------------------------------------------------------
# 모든 성분은 TARGET_SCALE로 나눈 단위에서 Huber(SmoothL1)로 계산함. TARGET_SCALE은 "달성 가능한 오차" 기준이라
# 기준선 성능에서 각 항이 비슷한 크기가 되고, 그 위에서 아래 가중치가 우선순위를 나타냄.
# 0.7 / 0.2 / 0.1은 논문에서 물려받은 값이고 한 번도 유도된 적이 없음. scale을 이렇게 맞추는 목적은 이 가중치를 나중에 바꿨을 때
# 효과를 예측할 수 있게 하는 것임.
LOSS_W_POS, LOSS_W_VEL, LOSS_W_COS = 0.7, 0.2, 0.1  # 위치 / 속도 / 방향(cosine) 가중치
LOSS_W_ACC = 0.05  # 가속도 가중치. 속도 항과 분리되어 있어 0으로 두면 속도 항은 그대로임. 0.05는 "작게"라는 취지로 정한 값이며 튜닝하지 않았음
LOSS_W_VISIBLE, LOSS_W_HIDDEN = 0.5, 0.5  # visible / hidden 부분집합 loss를 따로 평균낸 뒤 합치는 가중치 (튜닝하지 않은 값)
HUBER_BETA = 1.0  # TARGET_SCALE 단위(1.0 = 달성 가능한 오차 한 개)에서의 Huber 전환점
HEADING_MIN_SPEED = 0.5  # m/s. 정답 속력이 이보다 작으면 cosine(방향) 항에서 제외함.
# 근거: DFL 25Hz tracking 한 경기(DFL-MAT-J04034) 앞 9분, 한 팀 11명에서 1스텝 중심차분 속도의 방향이 +-5프레임 평활 속도의 방향과
# 벗어나는 정도를 쟀음(제가 측정). 0.5 m/s 미만에서 p90 편차가 18도를 넘고 이상에서는 10도 이하. 라벨은 Savitzky-Golay 평활이라 더 작을 수 있으나 재측정하지 않았음.
# 성분별 scale. 라벨 단위 기준(x, y는 field 길이/폭으로 나눈 값, v는 m/s, a는 m/s^2). 출처는 성분마다 다름:
#  x, y : 2.1 m (달성 가능한 오차). 독립적으로 측정된 두 값에 가깝다: (1) Phase 0 측정(coordinator 보고, 이 에이전트가 재측정하지 않음): 마지막 관측
#         위치를 그대로 쓸 때 중앙값 오차 1 s 2.05 m / 2 s 3.94 m / 5 s 8.56 m. (2) SCHEMA.md가 보고하는 360 vs tracking 위치 중앙값 오차 2.19 m
#         (visible 선수의 예측은 obs_xy에 대한 residual이므로 이 값이 하한). 2.1 m에서 기준선 오차(위치 2 m, 속도 1.9 m/s)의 위치 항과 속도 항이
#         가중치 비율(0.7 : 0.2)대로 나옴(smoke 6g).
#         알려진 근사: visible 선수의 달성 가능한 오차(약 2.19 m, 360 관측 잡음)와 hidden 선수의 것(대략 4 m, 전형적인 prev_dt에서)은 다르며,
#         한 scale로 둘 다 맞출 수 없음. loss가 두 부분집합을 따로 평균내므로 결함이 아니라 알려진 근사임. 부분집합별 scale은 두지 않음.
#         라벨 단위로 환산: x = 2.1/105 = 0.020000, y = 2.1/68 = 0.030882 (config.field_length=105, field_width=68).
#  vx, vy : 2.0 m/s (달성 가능한 오차). Phase 0 측정(coordinator 보고, 재측정 안 함): 360 sighting 중심차분의 성분별 RMSE 1.9~2.05 m/s, 실제 중앙값 속력 2.19 m/s.
#  ax, ay : 1.3, 1.1 m/s^2 (한계 표준편차이며 달성 가능한 오차가 아님). Phase 0에서 2차 차분 가속도가 정답과 상관 0.11 이하라 달성 가능한 오차 자체가 없음.
#         값은 제가 DFL-MAT-J04034 앞 9분 11명의 25Hz 위치를 5프레임 평활 후 0.4 s 차분해 구한 표준편차(ax 1.328, ay 1.141). Phase-2 라벨(Savitzky-Golay, 6 m/s^2 clip)에서 잰 값이 아님.
TARGET_SCALE = {"vx": 2.0, "vy": 2.0, "x": 2.1 / 105, "y": 2.1 / 68, "ax": 1.3, "ay": 1.1}
_COMPONENTS = {2: ("x", "y"), 4: ("vx", "vy", "x", "y"), 6: ("vx", "vy", "x", "y", "ax", "ay")}


def _masked_loss(preds, targets, comps):
    """preds, targets: (T, D) 한 부분집합의 token들. standardised Huber 항을 위치/속도/가속도/방향으로 묶어 가중합함."""
    scale = torch.tensor([TARGET_SCALE[c] for c in comps], device=preds.device, dtype=preds.dtype)
    huber = F.smooth_l1_loss(preds / scale, targets / scale, beta=HUBER_BETA, reduction="none")  # (T, D)
    pos_cols = [i for i, c in enumerate(comps) if c in ("x", "y")]
    loss_pos = huber[:, pos_cols].mean()
    if "vx" not in comps:
        return loss_pos
    vel_cols = [i for i, c in enumerate(comps) if c in ("vx", "vy")]
    loss = LOSS_W_POS * loss_pos + LOSS_W_VEL * huber[:, vel_cols].mean()
    acc_cols = [i for i, c in enumerate(comps) if c in ("ax", "ay")]
    if acc_cols:
        loss = loss + LOSS_W_ACC * huber[:, acc_cols].mean()
    # 방향: 속력이 작으면 atan2(vy, vx)가 의미 없으므로 정답 속력이 HEADING_MIN_SPEED를 넘는 token만 사용
    moving = targets[:, :2].norm(dim=1) > HEADING_MIN_SPEED
    if moving.any():
        loss = loss + LOSS_W_COS * (1 - F.cosine_similarity(preds[moving, :2], targets[moving, :2], dim=1)).mean()
    return loss


def balanced_loss(preds, targets, actor_global_index, visible, valid=None):
    """
    중앙 timestep의 (vx, vy, x, y, ax, ay) 예측에 대한 loss. visible과 hidden 선수를 따로 평균낸 뒤
    LOSS_W_VISIBLE / LOSS_W_HIDDEN으로 합침 (한쪽 부분집합이 비면 남은 쪽만 사용).
    라벨이 NaN인 token(뛰지 않는 선수)과 valid=False(padding)인 token은 제외함.
    preds, targets: (B, W, N, D)   visible, valid: (B, W, N)
    반환: (mae, loss, euclidean_dist[m], cosine_similarity)
    """

    B, W, N, feat_dim  = targets.shape
    preds = preds[:, W//2, :, :]
    targets = targets[:, W//2, :, :]
    visible = visible[:, W//2, :].bool()
    mask = ~torch.isnan(targets).any(dim=-1)
    if valid is not None:
        mask = mask & valid[:, W//2, :].bool()

    comps = _COMPONENTS[feat_dim]
    preds_masked, targets_masked, visible_masked = preds[mask], targets[mask], visible[mask]

    total, weight_sum = preds.sum() * 0.0, 0.0
    for subset, w in ((visible_masked, LOSS_W_VISIBLE), (~visible_masked, LOSS_W_HIDDEN)):
        if subset.any():
            total = total + w * _masked_loss(preds_masked[subset], targets_masked[subset], comps)
            weight_sum += w
    loss = total / weight_sum if weight_sum > 0 else total

    scale = torch.tensor(
        [config.field_length, config.field_width],
        device=preds_masked.device,
        dtype=preds_masked.dtype
    )
    xy_cols = [i for i, c in enumerate(comps) if c in ("x", "y")]
    euclidean_dist = torch.norm((preds_masked[:, xy_cols] - targets_masked[:, xy_cols]) * scale, dim=1)

    if "vx" in comps:
        moving = targets_masked[:, :2].norm(dim=1) > HEADING_MIN_SPEED
        cosine_similarity = F.cosine_similarity(preds_masked[moving, :2], targets_masked[moving, :2], dim=1)
    else:
        cosine_similarity = torch.tensor(float('nan'))  # 속도 벡터가 없으니 의미 없음

    mae = torch.abs(preds_masked - targets_masked)
    
    return mae.mean(dim=0), loss, euclidean_dist.mean(), cosine_similarity.mean()

class XGBoostModel():
    """Wrapper to unify constructor signature with other models."""
    def __init__(self, train_dataset, model_config=None, optimizer_params=None):
        self.model = XGBRegressor(**(model_config or {}))

class SetTransformerModel(pl.LightningModule):
    """SetTransformer 모델을 LightningModule로 감싸는 클래스"""

    def __init__(self, train_dataset, model_config: dict = None, optimizer_params: dict = None):
        super().__init__()

        sample=train_dataset[0]
        sample_input, sample_output, categorical_indices, actor_global_index = sample["features"], sample["labels"], sample["categorical_indices"], sample["actor_global_index_lst"]
        
        self.model = SetTransformer(
            dim_input=sample_input.shape[-1],
            dim_output=sample_output.shape[-1],
            num_outputs=sample_input.shape[-1],
            categorical_indices=categorical_indices,
            categorical_features=sample.get("categorical_features"),
            **model_config,

        )
        
        print("\n[모델 요약 정보]")
        total_params = 0
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            #print(f"{name:50} | shape: {str(tuple(param.shape))} | requires_grad: {param.requires_grad} | params: {param_count:,}")
        print(f"\n총 학습 가능한 파라미터 수: {total_params:,}\n")

        #self.criterion = torch.nn.MSELoss()  # ✅ MSE 손실 함수 추가
        self.criterion = balanced_loss
        self.save_hyperparameters(ignore=['train_dataset'])
        self.optimizer_params = optimizer_params if optimizer_params else {"lr": 1e-3}

    def forward(self, x, obs_xy, visible, valid=None):
        return self.model(x, obs_xy, visible, valid)  # ✅ 기본 forward 메서드

    def step(self, batch: Any):
        inputs, targets, actor_global_index = batch["features"], batch["labels"], batch["actor_global_index_lst"]
        obs_xy, visible, valid = batch["obs_xy"], batch["visible"], batch.get("valid", None)

        outputs = self.forward(inputs, obs_xy, visible, valid)

        mae, mse_loss, euclidean_dist, cosine_similarity = self.criterion(outputs, targets, actor_global_index, visible, valid)

        return mae, mse_loss, euclidean_dist, cosine_similarity
    
    def training_step(self, batch: Any, batch_idx: int):
        mae, mse_loss, euclidean_dist, cosine_similarity = self.step(batch)

        self.log("train_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_euclidean", euclidean_dist, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("train_cosine_sim", cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return mse_loss

    def validation_step(self, batch, batch_idx):
        mae, mse_loss, euclidean_dist, cosine_similarity = self.step(batch)

        self.log("val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_euclidean", euclidean_dist, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_cosine_sim", cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return mse_loss

    def test_step(self, batch, batch_idx):
        mae, mse_loss, euclidean_dist, cosine_similarity = self.step(batch)

        self.log("test_mse", mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_euclidean", euclidean_dist, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_cosine_sim", cosine_similarity, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return mse_loss
    
    def predict_step(self, batch, batch_idx):
        inputs, targets, actor_global_index = batch["features"], batch["labels"], batch["actor_global_index_lst"]
        obs_xy, visible, valid = batch["obs_xy"], batch["visible"], batch.get("valid", None)
        preds = self(inputs, obs_xy, visible, valid)  # 예측을 위한 forward 메서드 호출

        #return preds        
        return {
            "preds": preds,
            "labels": targets,
            "actor_global_index": actor_global_index,
            "visible": visible,
        }

    def configure_optimizers(self):
        """최적화 함수 설정"""
        return torch.optim.Adam(self.parameters(), **self.optimizer_params["optimizer_params"])

class AgentImputer(nn.Module):
    def __init__(self, dim_input=66, dim_hidden=100, output_size=2, categorical_indices=None):
        
        super().__init__()  
        self.hidden_layer_size = dim_hidden
        
        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.num_indices = [i for i in range(dim_input) if i not in self.categorical_indices]
        self.categorical_embeddings = nn.ModuleDict()
        
        for cat in config.categorical_features:
            num_classes = config.categorical_features[cat]  # 클래스 개수
            embed_dim = int(math.sqrt(num_classes))  # 임베딩 차원 설정
            self.categorical_embeddings[cat] = nn.Embedding(num_classes, embed_dim)
        # 기존 feature와 결합할 임베딩 차원 계산
        total_embedding_dim = sum([emb.embedding_dim for emb in self.categorical_embeddings.values()])
        new_input_dim = dim_input - len(self.categorical_indices) + total_embedding_dim  # 수정
        self.lstms = seq_lstm(new_input_dim, dim_hidden, dim_hidden)
        self.gcn = GCN(dim_hidden)
        # self.batch_size = batch_size
        self.relu = nn.ReLU()


    def forward(self, X, ts_list, edge_index):
        B, W, N, F = X.shape  # (Batch, Window, Players, Features)        

        # # 연속형 feature와 카테고리 feature 분리
        X_continuous = X[..., self.num_indices]  # 연속형 변수
        X_categorical = X[..., self.categorical_indices].long()  

        # 카테고리컬 변수 임베딩 적용
        embedded_features = []
        for i, cat in enumerate(sorted(config.categorical_features.keys())):  # 순서 보장
            mask = X_categorical[..., i] > 0  # 0보다 큰 값만 선택
            X_valid = X_categorical[..., i] * mask  # 0인 값들은 그대로 유지 (곱하기)
            embedded = torch.zeros_like(self.categorical_embeddings[cat](torch.zeros_like(X_valid)))  # 기본 0값 설정
            if mask.any():  # 값이 있는 경우만 임베딩 수행
                embedded[mask] = self.categorical_embeddings[cat](X_valid[mask] - 1)  # 올바르게 매칭된 임베딩 적용

            embedded_features.append(embedded)

        # 임베딩된 feature들을 결합
        if embedded_features:
            X_embedded = torch.cat(embedded_features, dim=-1)
            X = torch.cat([X_continuous, X_embedded], dim=-1)

        outputs = torch.cat([self.lstms(X[..., i, :],ts_list[..., i]) for i in range(40)],dim=1)
        outputs = outputs.reshape(B, N, -1)
        gcn_outputs = self.gcn(outputs, edge_index)
        return gcn_outputs

if __name__ =="__main__":
    pass
    # batch_size = 32
    # window = 5
    # num_players = 40
    # feature_dim = 15
    # output_dim = 4  # x, y 좌표 예측

    # # 가짜 입력 데이터 생성
    # fake_input = torch.randn(batch_size, window, num_players, feature_dim)

    # # 모델 초기화
    # model = SetTransformer(feature_dim, num_players, output_dim)

    # # Forward pass 테스트
    # output = model(fake_input)

    # print("Model Output Shape:", output.shape)  # 예상: (batch_size, 40, 2)
