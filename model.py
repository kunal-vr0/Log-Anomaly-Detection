import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from math import sqrt
from torch.nn.utils import weight_norm
import re
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader
out_dim = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

#Token Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

#Data Embedding
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

#Triangular Causal Mask
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

#Anomaly Attention
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

#Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma

#Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma

#Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

#Anomaly Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]

#Loss function
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

# getting timestamps from raw logs
def get_timestamp(text):
    pattern = r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+'
    timestamp = {}
    time = re.search(pattern, text)
    if time:
        x = time.group()
    x = x.split('-')
    timestamp['month'] = int(x[1])
    timestamp['date'] = int(x[2])
    y = x[3].split('.')
    timestamp['hour'] = int(y[0])
    timestamp['min'] = int(y[1])
    return timestamp

#get time difference between logs
# ((month1*30 + date1)*24 + hour1)*60 + min1
def diff(ref, temp):
    minutes = (((temp['month']*30 + temp['date'])*24 + temp['hour'])*60 + temp['min']) - ref
    if minutes < 0:
        minutes = -1
    return minutes

#extract input window based on time window
data_path = '/content/drive/MyDrive/LLM_cx/Anomaly-Transformer/dataset/My_BGL'
def get_window(window, data_path):
    for_Ref = 1
    log_window = []
    with open(data_path+'/validation.txt') as logs:
        for log in logs:
            if for_Ref:
                ref = get_timestamp(str(log))
                ref = ((ref['month']*30 + ref['date'])*24 + ref['hour'])*60 + ref['min']
                for_Ref = 0
            temp = get_timestamp(str(log))
            period = diff(ref, temp)
            if period < window:
                log_window.append(log)
            else:
                break
    return log_window

#Data Loader
class SegLoader(object):
    def __init__(self, data_path, window_len, win_size, step):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.win_len = window_len
        test_data = pd.read_csv(data_path + '/test_tf_50.csv')
        test_data = test_data[:self.win_len]
        test_data = np.nan_to_num(test_data)
        self.scaler.fit(test_data)
        self.test = self.scaler.transform(test_data)
        #print("test:", self.test.shape)

    def __len__(self):
        return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.test[index:index + self.win_size])


#making prediction
def predict(window):
  #------defining parameters----------
  temperature = 50
  criterion = nn.MSELoss(reduce=False)
  win_size = 100
  anormly_ratio = 1
  batch_size = 1024
  win_size = 100
  step = 100
  #---------loading data--------------
  data_path = '/content/drive/MyDrive/LLM_cx/Anomaly-Transformer/dataset/My_BGL'
  log_window = get_window(window)  # gets raw log entries form the defined window size in a list
  window_len = len(log_window)
  dataset = SegLoader(data_path, window_len, win_size, step)
  data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  print('window_len: ', window_len)
  #------------determining threshold-----------------
  # (1) find the threshold
  attens_energy = []
  for i, input_data in enumerate(data_loader):
    input = input_data.float().to(device)
    output, series, prior, _ = model(input)
    loss = torch.mean(criterion(input, output), dim=-1)

    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
      if u == 0:
        series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)).detach()) * temperature
        prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)),series[u].detach()) * temperature
      else:
        series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)).detach()) * temperature
        prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)),series[u].detach()) * temperature
    # Metric
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    attens_energy.append(cri)

  attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
  test_energy = np.array(attens_energy)
  combined_energy = test_energy
  #combined_energy = np.concatenate([train_energy, test_energy], axis=0)
  thresh = np.percentile(combined_energy, 100 - anormly_ratio)
  #print("Threshold :", thresh)

  #-----------------creating predictions----------------
  # (2) evaluation on the given dataset
  attens_energy = []
  for i, input_data in enumerate(data_loader):
    input = input_data.float().to(device)
    output, series, prior, _ = model(input)

    loss = torch.mean(criterion(input, output), dim=-1)

    series_loss = 0.0
    prior_loss = 0.0
    for u in range(len(prior)):
      if u == 0:
        series_loss = my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)).detach()) * temperature
        prior_loss = my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)),series[u].detach()) * temperature
      else:
        series_loss += my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)).detach()) * temperature
        prior_loss += my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,win_size)),series[u].detach()) * temperature
    metric = torch.softmax((-series_loss - prior_loss), dim=-1)
    cri = metric * loss
    cri = cri.detach().cpu().numpy()
    attens_energy.append(cri)

  attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
  test_energy = np.array(attens_energy)

  pred = (test_energy > thresh).astype(int)
  op_win = 5
  peaks = np.where(pred == 1)
  end = pred.shape[0]
  for i in peaks[0]:
    win_open = i-op_win
    win_close = i+op_win
    if win_open < 0:
      win_open = 0
    if win_close > end:
      win_close = end-1
    pred[win_open:win_close] = 1
  loc = np.where(pred == 1)
  print('anomalies: ', len(loc[0]))
  anomaly = [log_window[i] for i in loc[0]]

  return anomaly


