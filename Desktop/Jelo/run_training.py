#!/usr/bin/env python3
"""
Simple script to start training the commodity export prediction model
"""

import subprocess
import sys
import os

def main():
    # Change to the Predictor directory
    os.chdir(r"c:\Users\osmon\Desktop\Predictor")
    
    # Run the training script with default parameters
    cmd = [
        sys.executable, "train.py",
        "--data_path", r"..\custom_data\export\Illinois.csv",
        "--epochs", "5",  # Just a few epochs for testing
        "--d_model", "64",  # Smaller model for faster testing
        "--n_layers", "2",
        "--sequence_length", "6"
    ]
    
    print("Starting commodity export prediction training...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Training timed out after 5 minutes")
    except Exception as e:
        print(f"Error running training: {e}")

if __name__ == "__main__":
    main()
