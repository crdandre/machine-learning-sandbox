"""
Notes:

1. BCEWithLogitsLoss?
--> Commonly used for binary classification problems, measures distance from true binary value
--> sigmoid + binary cross entropy

2. What is binary cross-entropy?
--> BCE = -(y * log(p) + (1-y) * log(1-p))
        Where:
        y = true label (0 or 1)
        p = predicted probability (between 0 and 1)
    i.e., the difference between two prob. dists.
    read more on why this is the way it is...lol
    
    
3. Training vs Validation?
My first thought is why both are needed? 
In a one-epoch tranining run, the advantage of evaluating model performance on unseen samples is not there. Validation only offers batch-order bias mitigation and other mitigation due to differences in model.train() vs model.eval()
--> training forward-passes w/random dropout (sets activations to zero), measures loss, then backprops to update weights
--> eval does not do this, it forward-passes every eval sample and measures loss, no backprop
--> torch.no_grad() prevents gradient computation during the forward pass, which means backpropagation isn't possible because there are no gradients to propagate back.

"""


import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from carvana_dataset import CarvanaDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16
    EPOCHS = 2
    DATA_PATH = "./data"
    MODEL_SAVE_PATH = "./unet.pth"
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    train_dataset = CarvanaDataset(DATA_PATH)
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    
    # Use DistributedSampler for proper data distribution
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    model = UNet(in_channels=3, num_classes=1).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        model.train()
        train_running_loss = 0
        
        # Only show progress bar on main process
        if local_rank == 0:
            train_iter = tqdm(train_dataloader)
        else:
            train_iter = train_dataloader
            
        for idx, img_mask in enumerate(train_iter):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            
            y_pred = model(img)
            optimizer.zero_grad()
            
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        # Synchronize loss across processes
        train_loss = train_running_loss/(idx+1)
        
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            if local_rank == 0:
                val_iter = tqdm(val_dataloader)
            else:
                val_iter = val_dataloader
                
            for idx, img_mask in enumerate(val_iter):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                
                val_running_loss += loss.item()
                
            val_loss = val_running_loss/(idx+1)
            
        if local_rank == 0:
            print("-"*30)
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print("-"*30)

    if local_rank == 0:
        torch.save(model.module.state_dict(), MODEL_SAVE_PATH)  # Note: save the inner model
                
            
            
    
    
    
    