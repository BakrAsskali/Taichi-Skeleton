import numpy as np
import cv2
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, init_weights, GenNNSkeToImage


class Discriminator(nn.Module):
    """
    Discriminator Network:
    Classifies input images as 'Real' (from video) or 'Fake' (generated).
    Architecture: PatchGAN-style CNN (Convolutional Neural Network).
    Instead of outputting a single 0/1 for the whole image, it outputs a 4x4 grid
    of probabilities, determining if specific 'patches' of the image look real.
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # Base number of filters (standard for DCGAN architectures)
        ndf = 64

        self.main = nn.Sequential(
            # Layer 1: Input image (3 channels RGB x 64x64)
            # Conv2d: 3 inputs -> 64 filters, Kernel=4, Stride=2, Padding=1
            # Result: Downsamples image by half -> (64 filters x 32 x 32)
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyRelu allows weak gradients for negative values
            nn.Dropout2d(0.25),  # Dropout prevents the Discriminator from memorizing training data

            # Layer 2: Input (64 x 32 x 32)
            # Result: Downsamples -> (128 filters x 16 x 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),  # BatchNorm stabilizes learning by normalizing layer inputs
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Layer 3: Input (128 x 16 x 16)
            # Result: Downsamples -> (256 filters x 8 x 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Layer 4: Input (256 x 8 x 8)
            # Final Conv layer reduces to 1 channel map
            # Result: (1 channel x 4 x 4) -> This implies a PatchGAN output
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()  # Sigmoid squashes output to [0,1] range (Probability of Real)
        )
        # Apply custom weight initialization (usually mean=0, std=0.02 for GANs)
        self.apply(init_weights)
        print(self.main)

    def forward(self, input):
        return self.main(input)


class GenGAN():
    """
    Main GAN Controller Class.
    Manages the Generator (Skeleton -> Image) and Discriminator (Image -> Real/Fake),
    datasets, training loops, and inference.
    """

    def __init__(self, videoSke, loadFromFile=False):
        # Neural Networks
        self.netG = GenNNSkeToImage()  # The Generator Model
        self.netD = Discriminator()  # The Discriminator Model
        self.filename = 'data/Dance/DanceGenGAN.pth'  # Checkpoint path

        # Data Preprocessing:
        # 1. Resize to 64x64 (Model input size)
        # 2. CenterCrop ensures square aspect ratio
        # 3. ToTensor converts [0,255] pixels to [0,1] tensors
        # 4. Normalize converts [0,1] to [-1, 1] (Best for Tanh activation in Generator)
        tgt_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Initialize Dataset
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,  # Use 13 keypoints instead of full set to reduce noise
            target_transform=tgt_transform
        )

        # Initialize DataLoader for batch processing
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=16,  # Smaller batch sizes (16-32) often work better for GAN stability
            shuffle=True,  # Shuffle to break temporal correlations
            num_workers=2,
            pin_memory=True,  # Faster transfer to CUDA
            persistent_workers=True
        )

        # Device Selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if torch.cuda.is_available():
            print("CUDA available. Using GPU for training.")
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        # Load weights if requested
        if loadFromFile and os.path.isfile(self.filename):
            print(f"Loading model from {self.filename}")
            checkpoint = torch.load(self.filename)
            self.netG.load_state_dict(checkpoint['generator_state_dict'])
            self.netD.load_state_dict(checkpoint['discriminator_state_dict'])

    def train(self, num_epochs=200):
        # Define loss functions
        criterion_adv = nn.BCELoss()
        criterion_l1 = nn.L1Loss()

        # Setup optimizers
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # --- FIX: Removed 'verbose=True' ---
        schedulerD = optim.lr_scheduler.ReduceLROnPlateau(
            optimizerD, mode='min', factor=0.5, patience=15
        )
        schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
            optimizerG, mode='min', factor=0.5, patience=15
        )

        print("Starting Training Loop...")
        try:
            for epoch in range(num_epochs):
                errD_epoch = 0.0
                errG_epoch = 0.0
                D_x_epoch = 0.0
                D_G_z_epoch = 0.0
                n_batch = 0

                for i, (skeletons, real_images) in enumerate(self.dataloader):
                    batch_size = real_images.size(0)
                    real_images = real_images.to(self.device)
                    skeletons = skeletons.to(self.device)

                    real_label = (torch.rand(batch_size, 1, 4, 4) * 0.15 + 0.85).to(self.device)
                    fake_label = (torch.rand(batch_size, 1, 4, 4) * 0.15).to(self.device)

                    # Update Discriminator
                    self.netD.zero_grad()
                    output_real = self.netD(real_images)
                    errD_real = criterion_adv(output_real, real_label)
                    D_x = output_real.mean().item()

                    fake_images = self.netG(skeletons)
                    output_fake = self.netD(fake_images.detach())
                    errD_fake = criterion_adv(output_fake, fake_label)
                    D_G_z1 = output_fake.mean().item()

                    errD = errD_real + errD_fake

                    if D_x < 0.65 or D_G_z1 > 0.35:
                        errD.backward()
                        optimizerD.step()

                    # Update Generator
                    self.netG.zero_grad()
                    output_fake = self.netD(fake_images)
                    errG_adv = criterion_adv(output_fake, real_label)
                    errG_L1 = criterion_l1(fake_images, real_images)

                    l1_weight = max(30, 100 - epoch // 4)
                    errG = errG_adv + l1_weight * errG_L1
                    errG.backward()
                    D_G_z2 = output_fake.mean().item()
                    optimizerG.step()

                    errD_epoch += errD.item()
                    errG_epoch += errG.item()
                    D_x_epoch += D_x
                    D_G_z_epoch += (D_G_z1 + D_G_z2) / 2
                    n_batch += 1

                    if i % 32 == 0:
                        # --- OPTIONAL: Print Learning Rate manually here ---
                        current_lr_D = optimizerD.param_groups[0]['lr']
                        current_lr_G = optimizerG.param_groups[0]['lr']

                        print(f'[{epoch}/{num_epochs}][{i}/{len(self.dataloader)}] '
                              f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} '
                              f'L1: {l1_weight} LR_D: {current_lr_D:.6f}')

                        with torch.no_grad():
                            fake_img = self.dataset.tensor2image(fake_images[0].cpu())
                            real_img = self.dataset.tensor2image(real_images[0].cpu())
                            comparison = np.hstack([real_img, fake_img])
                            comparison = cv2.resize(comparison, (512, 256))
                            cv2.imshow('Real vs Fake', comparison)
                            cv2.waitKey(1)

                avg_errD = errD_epoch / n_batch
                avg_errG = errG_epoch / n_batch

                # Step the schedulers without verbose output
                schedulerD.step(avg_errD)
                schedulerG.step(avg_errG)

                if (epoch + 1) % 5 == 0:
                    print(f"Saving checkpoint at epoch {epoch + 1}")
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': self.netG.state_dict(),
                        'discriminator_state_dict': self.netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'schedulerG_state_dict': schedulerG.state_dict(),
                        'schedulerD_state_dict': schedulerD.state_dict(),
                    }, self.filename)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            cv2.destroyAllWindows()
            
    def generate(self, ske):
        """
        Inference method: Takes a skeleton, produces an image.
        """
        # Preprocess single skeleton (normalize, convert to tensor)
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(self.device)  # Add batch dimension

        self.netG.eval()  # Switch to evaluation mode (disable dropout, etc)
        with torch.no_grad():
            fake = self.netG(ske_t_batch)
            fake = fake.cpu()
            generated_image = self.dataset.tensor2image(fake[0])

        return generated_image


if __name__ == '__main__':
    # Parse command line argument for video file
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory =", os.getcwd())
    print("GenGAN: Filename =", filename)

    try:
        if not os.path.exists(filename):
            print(f"Error: Video file '{filename}' not found")
            sys.exit(1)

        # Extract skeletons from video
        targetVideoSke = VideoSkeleton(filename)

        # Initialize GAN with the video data
        # Set loadFromFile=True to resume training or use pre-trained weights
        gen = GenGAN(targetVideoSke, loadFromFile=False)

        try:
            print("\nStarting training (press Ctrl+C to stop)...")
            gen.train(200)  # Train for 200 epochs

            print("\nTesting generation (press 'q' to quit)...")
            # Loop through all skeletons in the video and generate frames
            for i in range(targetVideoSke.skeCount()):
                image = gen.generate(targetVideoSke.ske[i])
                image = cv2.resize(image, (256, 256))  # Resize for better visibility

                cv2.putText(image, f"Frame: {i}/{targetVideoSke.skeCount()}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)

                cv2.imshow('Generated Image', image)
                # Press 'q' to exit the playback loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nGeneration stopped by user")
                    break

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")

    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        cv2.destroyAllWindows()