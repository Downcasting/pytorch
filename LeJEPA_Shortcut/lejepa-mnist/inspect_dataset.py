import torch
import os
import matplotlib.pyplot as plt

def inspect_dataset(split='train'):
    data_path = f'./data/colored_mnist_{split}.pt'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    # Load the dictionary
    dataset = torch.load(data_path)
    
    # Access components
    images = dataset['images']
    targets = dataset['targets'] # The digit label (0-9)
    colors = dataset['colors']   # The color label (0-9)
    
    print(f"Loaded {split} dataset:")
    print(f"  Images shape: {images.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Colors shape: {colors.shape}")
    
    # Example: Check first 5 samples
    print("\nFirst 5 samples:")
    for i in range(5):
        print(f"  Index {i}: Digit Label = {targets[i].item()}, Color Label = {colors[i].item()}")

if __name__ == '__main__':
    inspect_dataset('train')
    inspect_dataset('test')
