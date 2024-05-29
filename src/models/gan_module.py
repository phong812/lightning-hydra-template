from typing import Any, Dict, Tuple

import wandb
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from lightning import LightningModule
import torch.nn as nn
from src.models.components.gan_component import *

class GAN(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Adam,
        #compile: bool,
        #net: torch.nn.Module,
        latent_dim = 100,
        ):
        
        super().__init__()
        
        # networks
        #self.net = net
        #mnist_shape = tuple([1, 28, 28])
        
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()

        self.validation_z = torch.randn(1, latent_dim)

        self.example_input_array = torch.zeros(2, latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        
        # print(imgs.shape[0])
        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if batch_idx % 2 == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            
            # sample_imgs = self.generated_imgs[:3]
            # real_sample_imgs = (sample_imgs * 255).type(torch.uint8)
            # self.logger.experiment.log({"generated_images": [wandb.Image(real_sample_imgs)]})
            
            # sample_imgs = self.generated_imgs[:5]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.log({"generated_images": [wandb.Image(grid)]}) 

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            # print("g_loss: ", g_loss)
            self.log("train/loss", g_loss, on_step=False, on_epoch=True, prog_bar=True) 
            return g_loss
            # return g_loss
            # tqdm_dict = {'g_loss': g_loss}
            # output = {
            #     'loss': g_loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # }
            # return output
            
            
        # train discriminator
        else:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("train/loss", d_loss, on_step=False, on_epoch=True, prog_bar=True)
            return d_loss
            # print("d_loss: ", d_loss)
            # return d_loss
            # tqdm_dict = {'d_loss': d_loss}
            # output = {
            #     'loss': d_loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # }
            # return output
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def plot_imgs(self):
        z = self.validation_z.to(self.device)

        # plt sampled images
        sample_imgs = self(z)

    def on_validation_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        # print(sample_imgs[0].shape)
        sample_imgs = sample_imgs.view(28, 28)
        # sample_imgs = (sample_imgs * 255).type(torch.uint8)
        self.logger.experiment.log({"generated_images": [wandb.Image(sample_imgs)]})
        
        
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('generated_images', grid, self.current_epoch) 

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True) 
        return loss
    
    # def test_step(self, batch, batch_idx):
        
        
if __name__ == "__main__":
    _ = GAN(None, None, None, None)
