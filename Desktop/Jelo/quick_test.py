"""
Simplified test to verify the commodity export prediction setup
"""

import os
try:
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    print("✓ All required packages imported successfully")
    
    # Test data loading
    data_path = r"c:\Users\osmon\Desktop\custom_data\export\Illinois.csv"
    print(f"✓ Checking data file: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found at {data_path}")
        exit(1)
    
    # Load and check data
    df = pd.read_csv(data_path, skiprows=1, header=1)
    print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['Time', 'Commodity', 'Country', 
                    'Containerized Vessel Total Exports Value ($US)', 
                    'Containerized Vessel Total Exports SWT (kg)']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        exit(1)
    
    print("✓ All required columns present")
    
    # Test basic data processing
    df = df.dropna()
    print(f"✓ After dropping NaN: {len(df)} rows")
    
    # Parse dates
    df['DateTime'] = pd.to_datetime(df['Time'], format='%b-%y', errors='coerce')
    df = df.dropna(subset=['DateTime'])
    print(f"✓ After parsing dates: {len(df)} rows")
    
    # Check unique values
    print(f"✓ Unique commodities: {df['Commodity'].nunique()}")
    print(f"✓ Unique countries: {df['Country'].nunique()}")
    print(f"✓ Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    
    print("\n🎉 Basic data validation passed! The model should work with this data.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install missing packages with: pip install torch pandas scikit-learn")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
