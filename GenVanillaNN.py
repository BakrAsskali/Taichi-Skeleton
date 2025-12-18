import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from torch.cuda.amp import autocast, GradScaler

torch.set_default_dtype(torch.float32)
# Number of channels in the training images
nc = 3
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 200
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset:
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image


    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output


# Weight initialization function
def init_weights(m):
    """Initialize network weights using the DCGAN strategy"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv layers: mean=0.0, std=0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm layers: mean=1.0, std=0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        # Generator network: skeleton coordinates -> RGB image
        self.main = nn.Sequential(
            # Layer 1: Input (26 joints) to feature maps
            # Input: (26) x 1 x 1 -> Output: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(26, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),  # Normalize features
            nn.ReLU(True),  # Non-linear activation

            # Layer 2: First upsampling
            # Input: (ngf*8) x 4 x 4 -> Output: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Layer 3: Second upsampling
            # Input: (ngf*4) x 8 x 8 -> Output: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Layer 4: Third upsampling
            # Input: (ngf*2) x 16 x 16 -> Output: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Layer 5: Final upsampling to image
            # Input: (ngf) x 32 x 32 -> Output: (3) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
        # Initialize weights using DCGAN strategy
        self.apply(init_weights)
        print(self.main)

    def forward(self, input):
        # Pass skeleton through generator network
        return self.main(input)


class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        # Input is RGB image of drawn skeleton (3 channels x 64 x 64)
        self.main = nn.Sequential(
            # 1. Start with 3-channel input (RGB drawn skeleton)
            # Downsample from 64x64 to 32x32
            nn.Conv2d(3, ngf, 4, 2, 1, bias=False),  # -> (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 2. Further downsample to capture skeleton structure
            # 32x32 to 16x16
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),  # -> (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 3. Compress to bottleneck
            # 16x16 to 8x8
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),  # -> (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 4. Start upsampling - expand features
            # 8x8 to 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # -> (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 5. Continue upsampling
            # 16x16 to 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # -> (ngf) x 32 x 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 6. Final upsampling to output size
            # 32x32 to 64x64 with 3 channels
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # -> (3) x 64 x 64
            nn.Tanh()  # Output range [-1, 1]
        )
        self.apply(init_weights)
        print(self.main)

    def forward(self, input):
        return self.main(input)


class GenVanillaNN():
    """Generator network for pose transfer, simpler version without discriminator
    Used as a baseline approach in 'Everybody Dance Now'
    """

    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        # Setup network architecture based on mode
        if optSkeOrImage == 1:
            # Direct skeleton coordinates to image
            self.netG = GenNNSkeToImage()
            self.filename = 'data/Dance/DanceVanillaFromSke.pth'
            src_transform = None
        else:
            # Skeleton drawing to image
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([
                SkeToImageTransform(64),  # Draw skeleton on 64x64 image
                transforms.ToTensor(),  # Convert to tensor
            ])
            self.filename = 'data/Dance/DanceVanillaFromSkeim.pth'

        # Setup data transforms for target images
        tgt_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Initialize dataset and dataloader
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            target_transform=tgt_transform,
            source_transform=src_transform
        )

        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        # Load pretrained model if requested
        if loadFromFile and os.path.isfile(self.filename):
            checkpoint = torch.load(self.filename)
            self.netG.load_state_dict(checkpoint['model_state_dict'])

    def train(self, n_epochs=20):
        """Train the generator network using L1 loss"""
        # Setup device (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            print("CUDA available. Using GPU for training.")
            self.netG = self.netG.cuda()

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.netG.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        criterion = nn.L1Loss()

        print(f"Starting training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for i, (skeletons, real_images) in enumerate(self.dataloader):
                # Move data to device
                skeletons = skeletons.to(device, non_blocking=True)
                real_images = real_images.to(device, non_blocking=True)

                ############################
                # Train Generator
                ############################
                self.netG.zero_grad()

                # Generate images
                fake_images = self.netG(skeletons)

                # Calculate L1 loss
                loss = criterion(fake_images, real_images)

                # Optimize
                loss.backward()
                optimizer.step()

                # Track loss
                epoch_loss += loss.item()

                # Clear CUDA cache if needed
                if i % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Print progress
                if i % 64 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss: {loss.item():.4f} '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Update learning rate
            avg_loss = epoch_loss / len(self.dataloader)
            scheduler.step(avg_loss)

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.netG.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                }, self.filename)

    def generate(self, ske):
        """Generate an image from a skeleton"""
        # Get current device
        device = next(self.netG.parameters()).device

        # Prepare input
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(device)

        # Generate image
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(ske_t_batch)
            fake = fake.cpu()
            generated_image = self.dataset.tensor2image(fake[0])

        return generated_image


if __name__ == '__main__':
    # Default values
    filename = "data/taichi1.mp4"
    load_model = False
    opt_mode = 2  # 1: skeleton, 2: skeleton image

    # Get arguments from command line
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        load_model = sys.argv[2].lower() == "true"
    if len(sys.argv) > 3:
        opt_mode = int(sys.argv[3])

    print("GenVanillaNN: Current Working Directory =", os.getcwd())
    print("GenVanillaNN: Filename =", filename)
    print(f"Load model: {load_model}, Mode: {opt_mode}")

    try:
        # Load video and create model
        targetVideoSke = VideoSkeleton(filename)
        print("targetVideoSke", targetVideoSke)

        # Create and train model
        gen = GenVanillaNN(targetVideoSke, loadFromFile=load_model, optSkeOrImage=opt_mode)
        gen.train(200)  # Train for 200 epochs

        # Test the model
        print("Testing the model...")
        for i in range(targetVideoSke.skeCount()):
            image = gen.generate(targetVideoSke.ske[i])
            image = cv2.resize(image, (256, 256))
            cv2.imshow('Generated Image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()