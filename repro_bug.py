import torch
from torch.utils.data import DataLoader, Dataset
import os

class DummyDataset(Dataset):
    def __init__(self):
        # Large tensors to potentially trigger memory issues
        self.data = [{"image": torch.randn(1, 96, 96, 96), "label": torch.randint(0, 2, (1, 96, 96, 96))}] * 20
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def reproduce():
    print("Testing DataLoader with num_workers=2 and pin_memory=True...")
    
    ds = DummyDataset()
    loader = DataLoader(ds, batch_size=2, num_workers=2, pin_memory=True)
    
    try:
        for i, batch in enumerate(loader):
            print(f"Batch {i} loaded")
            if i >= 5: break
        print("Success with basic Dataset")
    except Exception as e:
        print(f"Failed with basic Dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
