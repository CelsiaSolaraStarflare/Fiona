import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse
from tqdm import tqdm
from datetime import datetime

# Helper functions
def _c2r(x):
    return torch.view_as_real(x)

def _r2c(x):
    return torch.view_as_complex(x)

class SSMKernelDiag(nn.Module):
    def __init__(self, H, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        self.H = H
        self.N = N

        log_dt_min = math.log(dt_min)
        log_dt_max = math.log(dt_max)
        self.log_step = nn.Parameter(torch.rand(self.H) * (log_dt_max - log_dt_min) + log_dt_min)

        # Initialize A as negative real + imaginary for stability
        A_real = -0.5 + torch.rand(self.H, self.N)
        self.A_real = nn.Parameter(A_real)

        self.A_imag = nn.Parameter(torch.rand(self.H, self.N))

        B_real = torch.ones(self.H, self.N)
        B_imag = torch.zeros(self.H, self.N)
        self.B_real = nn.Parameter(B_real)
        self.B_imag = nn.Parameter(B_imag)

        self.C = nn.Parameter(torch.normal(0, 0.5**0.5, (self.H, self.N, 2)))

        self.D = nn.Parameter(torch.ones(self.H))

        if lr is None:
            self.lr = [None] * 3
        else:
            self.lr = lr if isinstance(lr, list) else [lr] * 3

    def forward(self, L):
        dt = torch.exp(self.log_step)  # (H)
        A = -torch.exp(self.A_real) + 1j * self.A_imag  # (H N)
        B = self.B_real + 1j * self.B_imag  # (H N)
        C = _r2c(self.C)  # (H N)

        dtA = dt.unsqueeze(1) * A  # (H, N)
        dA = torch.exp(dtA)  # (H, N)
        dB = (dt.unsqueeze(1) * B) * (dA - 1) / A  # (H, N)

        # Create convolution kernel
        powers = torch.arange(L, device=dA.device, dtype=dA.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        k_terms = (C * dB).unsqueeze(2) * (dA.unsqueeze(2) ** powers)  # (H, N, L)
        k = k_terms.sum(dim=1).real * 2  # Sum over N dimension, (H, L)

        return k

    def setup_step(self):
        dt = torch.exp(self.log_step)
        A = -torch.exp(self.A_real) + 1j * self.A_imag
        B = self.B_real + 1j * self.B_imag
        self.dA = torch.exp(dt.unsqueeze(1) * A)
        self.dB = (dt.unsqueeze(1) * B)
        self.dC = _r2c(self.C)

class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, activation='gelu', lr=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed

        self.kernel = SSMKernelDiag(H=d_model, N=d_state, lr=lr)

        self.activation = nn.GELU() if activation == 'gelu' else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, u, state=None):
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        k = self.kernel(L=L)  # (H L)

        y = torch.fft.rfft(u, n=2*L) * torch.fft.rfft(k, n=2*L) 
        y = torch.fft.irfft(y, n=2*L)[..., :L]

        y = y + u * self.kernel.D.unsqueeze(0).unsqueeze(2)

        y = self.activation(y)

        y = self.dropout(y)

        y = self.output_linear(y.transpose(-1, -2)).transpose(-1, -2)  # Adjust if needed

        if not self.transposed: y = y.transpose(-1, -2)
        return y, state

parser = argparse.ArgumentParser(description='PyTorch S4 Training for Commodity Export Prediction')
parser.add_argument('--data_path', default='../custom_data/export/Illinois.csv', type=str, help='Path to the commodity export data')
parser.add_argument('--d_model', default=128, type=int)
parser.add_argument('--n_layers', default=6, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--prenorm', action='store_true', help='Prenorm')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--sequence_length', default=12, type=int, help='Number of time steps to look back')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print(f'==> Preparing commodity export data from {args.data_path}..')

class CommodityExportDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, sequence_length=12, train=True, test_split=0.2):
        self.sequence_length = sequence_length
        
        # Load and preprocess data
        df = pd.read_csv(data_path, skiprows=1, header=1)
        df = df.dropna()
        
        # Parse time column (format like 'Jul-16', 'Feb-20')
        df['DateTime'] = pd.to_datetime(df['Time'], format='%b-%y', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        df = df.sort_values(['Commodity', 'Country', 'DateTime'])
        
        # Clean and convert value/weight columns
        value_col = 'Containerized Vessel Total Exports Value ($US)'
        weight_col = 'Containerized Vessel Total Exports SWT (kg)'
        
        df[value_col] = df[value_col].astype(str).str.replace(',', '').replace('', '0')
        df[weight_col] = df[weight_col].astype(str).str.replace(',', '').replace('', '0')
        
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce').fillna(0)
        
        # Create encoders for categorical features
        self.commodity_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()
        
        df['commodity_encoded'] = self.commodity_encoder.fit_transform(df['Commodity'])
        df['country_encoded'] = self.country_encoder.fit_transform(df['Country'])
        
        # Create sequences for each commodity-country combination
        self.sequences = []
        self.targets_value = []
        self.targets_weight = []
        self.commodity_features = []
        self.country_features = []
        
        for commodity in df['commodity_encoded'].unique():
            for country in df['country_encoded'].unique():
                subset = df[(df['commodity_encoded'] == commodity) & 
                           (df['country_encoded'] == country)].copy()
                
                if len(subset) >= sequence_length + 1:
                    # Create time features
                    subset['month'] = subset['DateTime'].dt.month
                    subset['year'] = subset['DateTime'].dt.year
                    subset['quarter'] = subset['DateTime'].dt.quarter
                    
                    # Normalize values
                    scaler_value = StandardScaler()
                    scaler_weight = StandardScaler()
                    
                    subset['value_norm'] = scaler_value.fit_transform(subset[[value_col]])
                    subset['weight_norm'] = scaler_weight.fit_transform(subset[[weight_col]])
                    
                    # Create sequences
                    for i in range(len(subset) - sequence_length):
                        seq_data = subset.iloc[i:i+sequence_length]
                        target_data = subset.iloc[i+sequence_length]
                        
                        # Features: [month, year, quarter, value_norm, weight_norm]
                        sequence = np.column_stack([
                            seq_data['month'].values,
                            seq_data['year'].values,
                            seq_data['quarter'].values,
                            seq_data['value_norm'].values,
                            seq_data['weight_norm'].values
                        ])
                        
                        self.sequences.append(sequence)
                        self.targets_value.append(target_data[value_col])
                        self.targets_weight.append(target_data[weight_col])
                        self.commodity_features.append(commodity)
                        self.country_features.append(country)
        
        # Convert to numpy arrays
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets_value = np.array(self.targets_value, dtype=np.float32)
        self.targets_weight = np.array(self.targets_weight, dtype=np.float32)
        self.commodity_features = np.array(self.commodity_features, dtype=np.int64)
        self.country_features = np.array(self.country_features, dtype=np.int64)
        
        # Split data
        n_samples = len(self.sequences)
        indices = np.arange(n_samples)
        train_indices, test_indices = train_test_split(indices, test_size=test_split, random_state=42)
        
        if train:
            self.indices = train_indices
        else:
            self.indices = test_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        sequence = torch.tensor(self.sequences[actual_idx], dtype=torch.float32)
        commodity = torch.tensor(self.commodity_features[actual_idx], dtype=torch.long)
        country = torch.tensor(self.country_features[actual_idx], dtype=torch.long)
        value_target = torch.tensor(self.targets_value[actual_idx], dtype=torch.float32)
        weight_target = torch.tensor(self.targets_weight[actual_idx], dtype=torch.float32)
        
        return sequence, commodity, country, value_target, weight_target

def split_train_val(train, val_split=0.1):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(train, [train_len, len(train) - train_len], 
                                               generator=torch.Generator().manual_seed(42))
    return train, val

# Create datasets
trainset = CommodityExportDataset(args.data_path, sequence_length=args.sequence_length, train=True)
testset = CommodityExportDataset(args.data_path, sequence_length=args.sequence_length, train=False)
trainset, valset = split_train_val(trainset)

# Get dataset info
sample_seq, sample_commodity, sample_country, sample_value, sample_weight = trainset[0]
d_input = sample_seq.shape[1]  # Number of features per time step
n_commodities = len(trainset.commodity_encoder.classes_)
n_countries = len(trainset.country_encoder.classes_)

print(f"Input dimension: {d_input}")
print(f"Sequence length: {args.sequence_length}")
print(f"Number of commodities: {n_commodities}")
print(f"Number of countries: {n_countries}")
print(f"Training samples: {len(trainset)}")
print(f"Validation samples: {len(valset)}")
print(f"Test samples: {len(testset)}")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

class CommodityS4Model(nn.Module):
    def __init__(self, d_input, n_commodities, n_countries, d_model, n_layers, dropout, prenorm):
        super().__init__()
        self.prenorm = prenorm
        self.d_model = d_model
        
        # Input encoder for time series
        self.encoder = nn.Linear(d_input, d_model)
        
        # Embedding layers for categorical features
        self.commodity_embedding = nn.Embedding(n_commodities, d_model // 4)
        self.country_embedding = nn.Embedding(n_countries, d_model // 4)
        
        # S4 layers
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, d_state=64, dropout=dropout, transposed=True))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Feature fusion layer
        self.fusion = nn.Linear(d_model + d_model // 4 + d_model // 4, d_model)
        
        # Output layers for value and weight prediction
        self.value_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.weight_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, commodity, country):
        # x shape: (batch, sequence_length, features)
        # Encode time series
        x = self.encoder(x)  # (batch, seq_len, d_model)
        x = x.transpose(-1, -2)  # (batch, d_model, seq_len)
        
        # Process through S4 layers
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = x + z
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        x = x.transpose(-1, -2)  # (batch, seq_len, d_model)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Get embeddings
        commodity_emb = self.commodity_embedding(commodity)  # (batch, d_model//4)
        country_emb = self.country_embedding(country)  # (batch, d_model//4)
        
        # Fuse features
        combined = torch.cat([x, commodity_emb, country_emb], dim=1)
        fused = self.fusion(combined)
        
        # Predict value and weight
        value_pred = self.value_decoder(fused)
        weight_pred = self.weight_decoder(fused)
        
        return value_pred.squeeze(-1), weight_pred.squeeze(-1)

model = CommodityS4Model(d_input, n_commodities, n_countries, args.d_model, args.n_layers, args.dropout, args.prenorm)
model.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['loss']  # Now using loss instead of accuracy
    start_epoch = checkpoint['epoch']

criterion_value = nn.MSELoss()
criterion_weight = nn.MSELoss()

def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not any(k in str(id(p)) for k in ["A_real", "log_step"])]
    ssm_params = [p for p in all_parameters if any(k in str(id(p)) for k in ["A_real", "log_step"])]
    param_groups = [
        {'params': params, 'weight_decay': weight_decay, 'lr': lr},
        {'params': ssm_params, 'weight_decay': 0.0, 'lr': 0.001}
    ]
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler

optimizer, scheduler = setup_optimizer(model, args.lr, args.weight_decay, args.epochs)

def train(epoch, dataloader):
    model.train()
    train_loss = 0
    total_value_loss = 0
    total_weight_loss = 0
    total = 0
    pbar = tqdm(enumerate(dataloader))
    for batch_idx, (sequences, commodities, countries, value_targets, weight_targets) in pbar:
        sequences, commodities, countries = sequences.to(device), commodities.to(device), countries.to(device)
        value_targets, weight_targets = value_targets.to(device), weight_targets.to(device)
        
        optimizer.zero_grad()
        value_outputs, weight_outputs = model(sequences, commodities, countries)
        
        loss_value = criterion_value(value_outputs, value_targets)
        loss_weight = criterion_weight(weight_outputs, weight_targets)
        loss = loss_value + loss_weight
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total_value_loss += loss_value.item()
        total_weight_loss += loss_weight.item()
        total += value_targets.size(0)
        
        pbar.set_description(
            'Batch Idx: (%d/%d) | Total Loss: %.3f | Value Loss: %.3f | Weight Loss: %.3f' %
            (batch_idx, len(dataloader), train_loss / (batch_idx + 1), 
             total_value_loss / (batch_idx + 1), total_weight_loss / (batch_idx + 1))
        )

def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    total_value_loss = 0
    total_weight_loss = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (sequences, commodities, countries, value_targets, weight_targets) in pbar:
            sequences, commodities, countries = sequences.to(device), commodities.to(device), countries.to(device)
            value_targets, weight_targets = value_targets.to(device), weight_targets.to(device)
            
            value_outputs, weight_outputs = model(sequences, commodities, countries)
            
            loss_value = criterion_value(value_outputs, value_targets)
            loss_weight = criterion_weight(weight_outputs, weight_targets)
            loss = loss_value + loss_weight
            
            eval_loss += loss.item()
            total_value_loss += loss_value.item()
            total_weight_loss += loss_weight.item()
            total += value_targets.size(0)
            
            pbar.set_description(
                'Batch Idx: (%d/%d) | Total Loss: %.3f | Value Loss: %.3f | Weight Loss: %.3f' %
                (batch_idx, len(dataloader), eval_loss / (batch_idx + 1),
                 total_value_loss / (batch_idx + 1), total_weight_loss / (batch_idx + 1))
            )
    
    if checkpoint:
        avg_loss = eval_loss / len(dataloader)
        if avg_loss < best_acc or best_acc == 0:  # Lower loss is better
            state = {
                'model': model.state_dict(),
                'loss': avg_loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = avg_loss

for epoch in range(start_epoch, args.epochs):
    print(f'\nEpoch: {epoch}')
    train(epoch, trainloader)
    eval(epoch, valloader)
    eval(epoch, testloader, checkpoint=True)
    scheduler.step()