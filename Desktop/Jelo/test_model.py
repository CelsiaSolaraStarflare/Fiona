#!/usr/bin/env python3
"""
Test script to verify the commodity export prediction model works
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import pandas as pd
import numpy as np
from train import CommodityExportDataset, CommodityS4Model

def test_dataset():
    print("Testing CommodityExportDataset...")
    
    # Test with the actual data path
    data_path = "../custom_data/export/Illinois.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return False
    
    try:
        dataset = CommodityExportDataset(data_path, sequence_length=6, train=True)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test getting a sample
        if len(dataset) > 0:
            sequence, commodity, country, value_target, weight_target = dataset[0]
            print(f"Sample shapes - Sequence: {sequence.shape}, Commodity: {commodity.shape}, Country: {country.shape}")
            print(f"Targets - Value: {value_target.item():.2f}, Weight: {weight_target.item():.2f}")
            
            # Test dataset info
            print(f"Number of commodities: {len(dataset.commodity_encoder.classes_)}")
            print(f"Number of countries: {len(dataset.country_encoder.classes_)}")
            
            return True, dataset
        else:
            print("Dataset is empty")
            return False, None
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return False, None

def test_model(dataset):
    print("\nTesting CommodityS4Model...")
    
    try:
        # Get model parameters from dataset
        sample_seq, sample_commodity, sample_country, sample_value, sample_weight = dataset[0]
        d_input = sample_seq.shape[1]
        n_commodities = len(dataset.commodity_encoder.classes_)
        n_countries = len(dataset.country_encoder.classes_)
        
        # Create model
        model = CommodityS4Model(
            d_input=d_input,
            n_commodities=n_commodities,
            n_countries=n_countries,
            d_model=64,  # Smaller for testing
            n_layers=2,  # Fewer layers for testing
            dropout=0.1,
            prenorm=False
        )
        
        print(f"Model created successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        sequences = torch.stack([dataset[i][0] for i in range(min(batch_size, len(dataset)))])
        commodities = torch.stack([dataset[i][1] for i in range(min(batch_size, len(dataset)))])
        countries = torch.stack([dataset[i][2] for i in range(min(batch_size, len(dataset)))])
        
        model.eval()
        with torch.no_grad():
            value_pred, weight_pred = model(sequences, commodities, countries)
            print(f"Forward pass successful - Value predictions: {value_pred.shape}, Weight predictions: {weight_pred.shape}")
            print(f"Sample predictions - Value: {value_pred[0].item():.2f}, Weight: {weight_pred[0].item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Starting model tests...\n")
    
    # Test dataset
    dataset_success, dataset = test_dataset()
    if not dataset_success:
        print("Dataset test failed, aborting.")
        return
    
    # Test model
    model_success = test_model(dataset)
    if not model_success:
        print("Model test failed.")
        return
    
    print("\nAll tests passed! The model is ready for training.")

if __name__ == "__main__":
    main()
