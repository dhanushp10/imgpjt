import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

# ============================
# Dataset Class
# ============================
class HybridNoisyCleanDataset(Dataset):
    def __init__(self, format_type, noisy_dir=None, clean_dir=None, root_dir=None, transform=None):
        self.format_type = format_type
        self.transform = transform

        if format_type == 'flat':
            assert noisy_dir and clean_dir, "Flat format requires noisy_dir and clean_dir"
            self.pairs = self._load_flat_pairs(noisy_dir, clean_dir)
        elif format_type == 'folder':
            assert root_dir, "Folder format requires root_dir"
            self.pairs = self._load_folder_pairs(root_dir)
        else:
            raise ValueError("format_type must be 'flat' or 'folder'")

    def _load_flat_pairs(self, noisy_dir, clean_dir):
        noisy_files = sorted(os.listdir(noisy_dir))
        clean_files = sorted(os.listdir(clean_dir))
        pairs = []
        for noisy_file in noisy_files:
            clean_name = noisy_file[len("noisy_"):] if noisy_file.startswith("noisy_") else noisy_file
            if clean_name in clean_files:
                pairs.append((os.path.join(noisy_dir, noisy_file), os.path.join(clean_dir, clean_name)))
        return pairs

    def _load_folder_pairs(self, root_dir):
        pairs = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if len(images) >= 2:
                noisy_path = os.path.join(folder_path, images[0])
                clean_path = os.path.join(folder_path, images[1])
                pairs.append((noisy_path, clean_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy_img = Image.open(noisy_path).convert("L")
        clean_img = Image.open(clean_path).convert("L")

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img

# ============================
# UNet-Style Autoencoder
# ============================
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ============================
# Transform
# ============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ============================
# Dataset Loading
# ============================
USE_FLAT_FORMAT = True

if USE_FLAT_FORMAT:
    dataset = HybridNoisyCleanDataset(
        format_type='flat',
        noisy_dir=r"C:\Users\dhanu\OneDrive\文档\4th sem all documents\imgPjt\Noisy_folder",
        clean_dir=r"C:\Users\dhanu\OneDrive\文档\4th sem all documents\imgPjt\Ground_truth",
        transform=transform
    )
else:
    dataset = HybridNoisyCleanDataset(
        format_type='folder',
        root_dir=r"C:\Users\dhanu\OneDrive\文档\4th sem all documents\imgPjt\Data",
        transform=transform
    )

# ============================
# Dataloader
# ============================
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ============================
# Loss Function
# ============================
def mixed_loss(output, target):
    mse = nn.functional.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return mse + 0.2 * ssim_loss

vgg = models.vgg16(pretrained=True).features[:9].eval()
vgg.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(output, target):
    output_rgb = output.repeat(1, 3, 1, 1)
    target_rgb = target.repeat(1, 3, 1, 1)
    return nn.functional.mse_loss(vgg(output_rgb), vgg(target_rgb))

def total_loss(output, target):
    return mixed_loss(output, target) + 0.05 * perceptual_loss(output, target)

# ============================
# Training
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for noisy_imgs, clean_imgs in dataloader:
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = total_loss(outputs, clean_imgs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# ============================
# Save Model
# ============================
torch.save(model.state_dict(), "denoising_unet_autoencoder.pth")

# ============================
# Visualize Results
# ============================
model.eval()
with torch.no_grad():
    noisy_imgs, clean_imgs = next(iter(dataloader))
    noisy_imgs = noisy_imgs.to(device)
    outputs = model(noisy_imgs)

    fig, axs = plt.subplots(3, 5, figsize=(12, 6))
    for i in range(min(5, noisy_imgs.size(0))):
        axs[0, i].imshow(noisy_imgs[i].cpu().squeeze(), cmap='gray')
        axs[0, i].set_title("Noisy")
        axs[1, i].imshow(outputs[i].cpu().squeeze(), cmap='gray')
        axs[1, i].set_title("Denoised")
        axs[2, i].imshow(clean_imgs[i].cpu().squeeze(), cmap='gray')
        axs[2, i].set_title("Clean")
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()