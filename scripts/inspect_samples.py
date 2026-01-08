#!/usr/bin/env python
import numpy as np

# Simple data inspection
sample_file = "/lotterlab/emily_torchxrayvision/outputs/gan_fast_fixed/samples/samples_epoch_10.npy"

print(f"🔍 Inspecting: {sample_file}")

try:
    # First try without allowing pickle
    try:
        data = np.load(sample_file)
        print(f"Regular load successful")
    except:
        # Try with pickle
        data = np.load(sample_file, allow_pickle=True)
        print(f"Pickle load successful")
    
    print(f"Data type: {type(data)}")
    
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    
    # If it's an object, try to access as item
    if hasattr(data, 'item'):
        try:
            item_data = data.item()
            print(f"Item data type: {type(item_data)}")
            
            if isinstance(item_data, dict):
                print(f"Dictionary keys: {list(item_data.keys())}")
                for key, value in item_data.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")
            
        except:
            print("Could not access as item")
    
except Exception as e:
    print(f"Error: {e}")
