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


import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from unet import UNet
from carvana_dataset import CarvanaDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()
    
    LEARNING_RATE = 3e-4
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATA_PATH = "./data"
    MODEL_SAVE_PATH = "./unet.pth"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_PATH, max_images=args.max_images)
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            
            y_pred = model(img)
            optimizer.zero_grad()
            
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        train_loss = train_running_loss/idx+1
        
        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                
                val_running_loss += loss.item()
                
            val_loss = val_running_loss/idx+1
            
        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                
            
            
    
    
    
    