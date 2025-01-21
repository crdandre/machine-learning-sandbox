import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

from carvana_dataset import CarvanaDataset
from unet import UNet

def pred_show_image_grid(data_path, model_pth, device, output_dir="inference_outputs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    
    # Use non-interactive Agg backend
    plt.switch_backend('Agg')
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
       fig.add_subplot(3, len(image_dataset), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.savefig(os.path.join(output_dir, "batch_inference.png"))
    plt.close()


def single_image_inference(image_pth, model_pth, device, output_dir="inference_outputs"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0]=0
    pred_mask[pred_mask > 0]=1

    # Use non-interactive Agg backend
    plt.switch_backend('Agg')
    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
    plt.savefig(os.path.join(output_dir, f"single_inference_{os.path.basename(image_pth)}"))
    plt.close()


if __name__ == "__main__":
    SINGLE_IMG_PATH = "./data/manual_test/0cdf5b5d0ce1_01.jpg"
    DATA_PATH = "./data"
    MODEL_PATH = "./models/unet.pth"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = "inference_outputs"
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device, output_dir)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device, output_dir)