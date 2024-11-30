import torch

model_path = '/storage/home/sqd5856/Desktop/default/protein_design/LatentMD/MAIN1115_checkpoints/ProAutoMD_epoch_2.pt'

try:
    model = torch.load(model_path)
    print("yes")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"p: {total_params}")
except Exception as e:
    print(f" error: {e}")

