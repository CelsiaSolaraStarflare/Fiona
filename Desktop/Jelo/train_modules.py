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

    def forward(self, u):
        """Forward pass to compute the kernel matrix"""
        # u shape: [batch * d_model, seq_len]
        batch_d_model, seq_len = u.shape
        
        step = torch.exp(self.log_step)  # [H]
        
        A = torch.complex(self.A_real, self.A_imag)  # [H, N]
        B = torch.complex(self.B_real, self.B_imag)  # [H, N]
        C = _r2c(self.C)  # [H, N]
        
        # Discretize system
        A_bar = torch.exp(A * step.unsqueeze(-1))  # [H, N]
        B_bar = (A_bar - 1) / A * B  # [H, N]
        
        # Create a simple kernel based on the discretized system
        # Just use the mean of the kernel for each dimension
        kernel_weights = torch.mean(torch.real(C * B_bar), dim=-1)  # [H]
        
        # Calculate how many full groups of H we have
        num_groups = batch_d_model // self.H
        remainder = batch_d_model % self.H
        
        if remainder != 0:
            # If there's a remainder, pad or truncate appropriately
            # For simplicity, let's just use a linear layer instead
            output = u * 0.5  # Simple scaling operation
        else:
            # Reshape u to [num_groups, H, seq_len]
            u_reshaped = u.view(num_groups, self.H, seq_len)
            
            # Apply kernel weights: multiply each channel by its corresponding weight
            output_reshaped = u_reshaped * kernel_weights.unsqueeze(0).unsqueeze(-1)
            
            # Reshape back to [batch * H, seq_len]
            output = output_reshaped.view(batch_d_model, seq_len)
        
        return output

class S4Layer(nn.Module):
    def __init__(self, d_model, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.H = d_model
        # Simplified S4-inspired layer using linear transformations
        self.projection = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.transposed = transposed

    def forward(self, u):
        # Input u should be [batch, seq_len, d_model]
        # Do NOT transpose - let's work with the expected shape
        
        # Apply a simple transformation that mimics S4 behavior
        y = self.projection(u)  # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        y = self.gelu(y)
        y = self.dropout(y)
            
        return y

class CommodityExportDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, sequence_length=12, train=True, test_split=0.2):
        self.sequence_length = sequence_length
        
        # Load data with proper structure for this CSV format
        # Skip first 2 rows (metadata), use row 3 as header
        df = pd.read_csv(data_path, skiprows=2, header=0)
        
        # Clean and prepare data
        df = df.dropna()
        
        print(f"CSV loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # The columns we expect from the CSV structure
        expected_cols = {
            'State': 'State',
            'Commodity': 'Commodity', 
            'Country': 'Country',
            'Time': 'Time',
            'Containerized Vessel Total Exports Value ($US)': 'Containerized Vessel Total Exports Value ($US)',
            'Containerized Vessel Total Exports SWT (kg)': 'Containerized Vessel Total Exports SWT (kg)'
        }
        
        # Verify all required columns exist
        missing_cols = []
        for expected_col in expected_cols.keys():
            if expected_col not in df.columns:
                missing_cols.append(expected_col)
        
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select and rename columns
        df_filtered = df[list(expected_cols.keys())].copy()
        
        # Clean monetary values (remove commas and handle empty values)
        value_col = 'Containerized Vessel Total Exports Value ($US)'
        weight_col = 'Containerized Vessel Total Exports SWT (kg)'
        
        if value_col in df_filtered.columns:
            # Remove commas and convert to float, handle missing values
            df_filtered[value_col] = (
                df_filtered[value_col].astype(str)
                .str.replace(',', '')
                .str.replace('nan', '0')
                .replace('', '0')
            )
            # Convert to numeric, setting errors to NaN
            df_filtered[value_col] = pd.to_numeric(df_filtered[value_col], errors='coerce').fillna(0)
        
        if weight_col in df_filtered.columns:
            # Remove commas and convert to float, handle missing values
            df_filtered[weight_col] = (
                df_filtered[weight_col].astype(str)
                .str.replace(',', '')
                .str.replace('nan', '0')
                .replace('', '0')
            )
            # Convert to numeric, setting errors to NaN
            df_filtered[weight_col] = pd.to_numeric(df_filtered[weight_col], errors='coerce').fillna(0)
        
        # Remove rows where both value and weight are 0 or NaN
        df_filtered = df_filtered[
            (df_filtered[value_col] > 0) | (df_filtered[weight_col] > 0)
        ].copy()
        
        print(f"After cleaning: {len(df_filtered)} rows remaining")
        
        # Process time features - handle format like "Jul-16", "Feb-20"
        if 'Time' in df_filtered.columns:
            try:
                # Convert time format like "Jul-16" to proper datetime
                df_filtered['Time'] = pd.to_datetime(df_filtered['Time'], format='%b-%y')
                df_filtered['Month'] = df_filtered['Time'].dt.month
                df_filtered['Year'] = df_filtered['Time'].dt.year
                df_filtered['Quarter'] = df_filtered['Time'].dt.quarter
                print("✓ Time processing successful")
            except Exception as e:
                print(f"Warning: Time processing failed: {e}")
                # Fallback: create dummy time features
                df_filtered['Month'] = 1
                df_filtered['Year'] = 2020
                df_filtered['Quarter'] = 1
        
        # Encode categorical variables
        self.commodity_encoder = LabelEncoder()
        self.country_encoder = LabelEncoder()
        
        if 'Commodity' in df_filtered.columns:
            df_filtered['Commodity_encoded'] = self.commodity_encoder.fit_transform(df_filtered['Commodity'])
        if 'Country' in df_filtered.columns:
            df_filtered['Country_encoded'] = self.country_encoder.fit_transform(df_filtered['Country'])
        
        print(f"✓ Encoded {len(self.commodity_encoder.classes_)} commodities and {len(self.country_encoder.classes_)} countries")
        
        # Normalize numerical features
        self.value_scaler = StandardScaler()
        self.weight_scaler = StandardScaler()
        
        value_col = 'Containerized Vessel Total Exports Value ($US)'
        weight_col = 'Containerized Vessel Total Exports SWT (kg)'
        
        if value_col in df_filtered.columns:
            # Only normalize non-zero values
            non_zero_values = df_filtered[df_filtered[value_col] > 0][value_col].values.reshape(-1, 1)
            if len(non_zero_values) > 0:
                self.value_scaler.fit(non_zero_values)
                df_filtered['Value_normalized'] = df_filtered[value_col].apply(
                    lambda x: self.value_scaler.transform([[x]])[0][0] if x > 0 else 0
                )
            else:
                df_filtered['Value_normalized'] = 0
        
        if weight_col in df_filtered.columns:
            # Only normalize non-zero values
            non_zero_weights = df_filtered[df_filtered[weight_col] > 0][weight_col].values.reshape(-1, 1)
            if len(non_zero_weights) > 0:
                self.weight_scaler.fit(non_zero_weights)
                df_filtered['Weight_normalized'] = df_filtered[weight_col].apply(
                    lambda x: self.weight_scaler.transform([[x]])[0][0] if x > 0 else 0
                )
            else:
                df_filtered['Weight_normalized'] = 0
        
        print("✓ Normalization completed")
        
        # Create sequences
        self.sequences = []
        self.commodities = []
        self.countries = []
        self.target_values = []
        self.target_weights = []
        
        # Sort by time
        df_filtered = df_filtered.sort_values('Time')
        
        # Group by commodity and country to create time series
        for (commodity, country), group in df_filtered.groupby(['Commodity_encoded', 'Country_encoded']):
            if len(group) > sequence_length:
                group = group.sort_values('Time')
                
                for i in range(len(group) - sequence_length):
                    # Create sequence of features (month, year, quarter, value, weight)
                    sequence_data = []
                    for j in range(i, i + sequence_length):
                        row = group.iloc[j]
                        sequence_data.append([
                            row['Month'], 
                            row['Year'], 
                            row['Quarter'],
                            row['Value_normalized'] if 'Value_normalized' in row else 0,
                            row['Weight_normalized'] if 'Weight_normalized' in row else 0
                        ])
                    
                    # Target is the next time step
                    target_row = group.iloc[i + sequence_length]
                    
                    self.sequences.append(sequence_data)
                    self.commodities.append(commodity)
                    self.countries.append(country)
                    self.target_values.append(target_row['Value_normalized'] if 'Value_normalized' in target_row else 0)
                    self.target_weights.append(target_row['Weight_normalized'] if 'Weight_normalized' in target_row else 0)
        
        # Convert to tensors
        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.commodities = torch.tensor(self.commodities, dtype=torch.long)
        self.countries = torch.tensor(self.countries, dtype=torch.long)
        self.target_values = torch.tensor(self.target_values, dtype=torch.float32)
        self.target_weights = torch.tensor(self.target_weights, dtype=torch.float32)
        
        # Train/test split
        total_samples = len(self.sequences)
        split_idx = int(total_samples * (1 - test_split))
        
        if train:
            self.sequences = self.sequences[:split_idx]
            self.commodities = self.commodities[:split_idx]
            self.countries = self.countries[:split_idx]
            self.target_values = self.target_values[:split_idx]
            self.target_weights = self.target_weights[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]
            self.commodities = self.commodities[split_idx:]
            self.countries = self.countries[split_idx:]
            self.target_values = self.target_values[split_idx:]
            self.target_weights = self.target_weights[split_idx:]
        
        print(f"Dataset created: {len(self.sequences)} samples ({'train' if train else 'test'})")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], self.commodities[idx], self.countries[idx], 
                self.target_values[idx], self.target_weights[idx])

class CommodityS4Model(nn.Module):
    def __init__(self, d_input, n_commodities, n_countries, d_model=128, n_layers=4, dropout=0.2, prenorm=False):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(d_input, d_model)
        
        # Embedding layers for categorical features
        self.commodity_embedding = nn.Embedding(n_commodities, d_model // 4)
        self.country_embedding = nn.Embedding(n_countries, d_model // 4)
        
        # S4 layers
        self.s4_layers = nn.ModuleList([
            S4Layer(d_model, dropout=dropout) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(d_model + d_model // 2, d_model)
        self.value_head = nn.Linear(d_model, 1)
        self.weight_head = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.prenorm = prenorm
        
    def forward(self, sequences, commodities, countries):
        # Project input sequences
        x = self.input_projection(sequences)  # [batch, seq_len, d_model]
        
        # Get embeddings
        commodity_emb = self.commodity_embedding(commodities)  # [batch, d_model//4]
        country_emb = self.country_embedding(countries)        # [batch, d_model//4]
        
        # Combine embeddings
        context_emb = torch.cat([commodity_emb, country_emb], dim=-1)  # [batch, d_model//2]
        
        # Add context to each time step
        context_expanded = context_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        # Pass through S4 layers
        for i, (s4_layer, layer_norm) in enumerate(zip(self.s4_layers, self.layer_norms)):
            if self.prenorm:
                x_normed = layer_norm(x)
                x = x + self.dropout(s4_layer(x_normed))
            else:
                x = x + self.dropout(s4_layer(x))
                x = layer_norm(x)
        
        # Take the last time step
        x = x[:, -1, :]  # [batch, d_model]
        
        # Concatenate with context
        x = torch.cat([x, context_emb], dim=-1)  # [batch, d_model + d_model//2]
        
        # Final projection
        x = self.output_projection(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        # Generate predictions
        value_pred = self.value_head(x).squeeze(-1)
        weight_pred = self.weight_head(x).squeeze(-1)
        
        return value_pred, weight_pred
