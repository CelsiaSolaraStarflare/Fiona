# Commodity Export Prediction with S4 Models

This is a modified version of the S4 model training script adapted for predicting Illinois commodity exports based on historical data.

## Overview

The original `train.py` has been modified to:

1. **Load commodity export data** instead of image datasets (CIFAR-10/MNIST)
2. **Process time series data** with features like time, commodity type, and country
3. **Predict export values and weights** using a regression approach instead of classification
4. **Use embeddings** for categorical features (commodities and countries)
5. **Handle sequential data** with configurable sequence lengths

## Key Changes Made

### Data Loading
- Replaced image dataset loaders with `CommodityExportDataset` class
- Handles CSV data with time series parsing (format: 'Jul-16', 'Feb-20', etc.)
- Creates sequences of historical data points for training
- Includes categorical encoders for commodities and countries

### Model Architecture
- Renamed `S4Model` to `CommodityS4Model`
- Added embedding layers for commodity and country features
- Changed output to dual regression heads (value and weight prediction)
- Modified input processing to handle time series features

### Loss Function
- Replaced CrossEntropyLoss with MSELoss for regression
- Dual loss calculation for both export value and weight predictions

### Training/Evaluation
- Updated metrics from accuracy to loss-based evaluation
- Modified progress display to show value and weight losses
- Changed checkpoint saving to use loss instead of accuracy

## Usage

### Basic Training
```bash
python train.py --data_path ../custom_data/export/Illinois.csv --epochs 50
```

### Custom Parameters
```bash
python train.py \
    --data_path ../custom_data/export/Illinois.csv \
    --epochs 80 \
    --d_model 128 \
    --n_layers 6 \
    --sequence_length 12 \
    --lr 0.01 \
    --dropout 0.2
```

### Command Line Arguments

- `--data_path`: Path to the commodity export CSV file
- `--d_model`: Model dimension (default: 128)
- `--n_layers`: Number of S4 layers (default: 6)
- `--dropout`: Dropout rate (default: 0.2)
- `--sequence_length`: Number of historical time steps to use (default: 12)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: Weight decay for optimizer (default: 0.01)
- `--epochs`: Number of training epochs (default: 80)
- `--prenorm`: Use pre-normalization
- `--resume`: Resume from checkpoint

## Data Format

The expected CSV format (after skipping metadata rows):
```
State,Commodity,Country,Time,Vessel Value ($US),Containerized Vessel Total Exports Value ($US),Vessel SWT (kg),Containerized Vessel Total Exports SWT (kg)
Illinois,01 Live Animals,Africa,Jul-16,"7,185","7,185",6,6
...
```

Key requirements:
- Time format: 'MMM-YY' (e.g., 'Jul-16', 'Feb-20')
- Numeric values may contain commas
- Missing values in containerized columns are handled

## Model Architecture

```
Input Sequence (batch, seq_len, features) 
    ↓
Linear Encoder → S4 Layers → Global Average Pooling
    ↓                           ↓
Categorical Embeddings ----→ Feature Fusion
    ↓
Dual Output Heads (Value & Weight Prediction)
```

## Features

The model uses the following features:
- **Temporal**: Month, year, quarter extracted from time strings
- **Sequential**: Historical export values and weights (normalized)
- **Categorical**: Commodity type and destination country (embedded)

## Testing

Run the quick test to verify setup:
```bash
python quick_test.py
```

This will check:
- Package imports
- Data file accessibility  
- Required columns presence
- Basic data processing

## Output

The model predicts:
1. **Export Value**: Dollar amount of exports
2. **Export Weight**: Weight in kilograms

Both predictions are made simultaneously using the shared S4 representation with separate output heads.

## Checkpoints

- Saved in `./checkpoint/ckpt.pth`
- Contains model state, loss, and epoch information
- Use `--resume` flag to continue training from checkpoint
