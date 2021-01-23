import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

!pip install utils
import utils

CUDA = True
batch_size = 128
CHANNELS = 3
Z_DIM = 100
HIDDENG = 64
X_DIM = 64
HIDDEND = 64
lr = 2e-4
seed = 2

CUDA = CUDA and torch.cuda.is_available()

np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
  torch.cuda.manual_seed(seed)


cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
        *self._blockG(Z_DIM, HIDDENG*8, 4, 1, 0),
        *self._blockG(HIDDENG*8, HIDDENG*4),
        *self._blockG(HIDDENG*4, HIDDENG*2),
        *self._blockG(HIDDENG*2, HIDDENG),
        nn.ConvTranspose2d(HIDDENG, CHANNELS, 4, 2, 1, bias=False),
        nn.Tanh())
  
  def _blockG(self,n_in, n_out, filters=4, stride=2, padding=1, bn=True):
    layers = [nn.ConvTranspose2d(n_in, n_out, filters, stride, padding, bias=False)]
    if bn:
      layers.append(nn.BatchNorm2d(n_out))
    layers.append(nn.ReLU(True))

    return layers
      
  def forward(self, input):
    return self.main(input)
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      *self._blockD(CHANNELS, HIDDEND, bn=False),
      *self._blockD(HIDDEND, HIDDEND*2),
      *self._blockD(HIDDEND*2, HIDDEND*4),
      *self._blockD(HIDDEND*4, HIDDEND*8),
      nn.Conv2d(HIDDEND * 8, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )
  
  def _blockD(self, n_in, n_out, kernel_s=4, stride=2, padding=1, bn=True):
    layers = [nn.Conv2d(n_in, n_out, kernel_s, stride, padding, bias=False)]
    if bn:
      layers.append(nn.BatchNorm2d(n_out))
    layers.append(nn.LeakyReLU(0.2,inplace=True))
    return layers

  def forward(self, x):
    x = self.main(x)
    return x.view(-1, 1).squeeze(1)
 
 def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)

def alreadytrained(state, files=None):
  netD = Discriminator().to(device)
  netG = Generator().to(device)
  if state == 1:
    modelG = torch.load(files[0])
    netG.load_state_dict(modelG)
    modelD = torch.load(files[1])
    netD.load_state_dict(modelD)
  else:
    netD.apply(weights_init)
    netG.apply(weights_init)
  return netD, netG

path2files = ["GENGANPaintings.pt", "DISGANPaintings.pt"]
netD, netG = alreadytrained(0)


criterion = nn.BCELoss()
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

from PIL import ImageFile
import PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset = datasets.ImageFolder(root="drive/MyDrive/Paintings",
                     transform=transforms.Compose(
                         [
                          transforms.Resize((X_DIM,X_DIM)),
                          transforms.ToTensor(),
                         ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)

fixed_noise = torch.randn(10, Z_DIM, 1, 1, device=device)
n_epochs = 100
gens_epochs = []
for epoch in range(n_epochs):
  for i, (X, y) in enumerate(dataloader):
    x_real = X.to(device)

    real_labels = torch.ones(x_real.size(0), device=device).float()
    fake_labels = torch.zeros(x_real.size(0), device=device).float()
    
    netD.zero_grad()

    y_real = netD(x_real)

    loss_D_real = criterion(y_real, real_labels)

    z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
    x_fake = netG(z_noise)
    y_fake = netD(x_fake.detach())
    loss_D_fake = criterion(y_fake, fake_labels)

    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optimD.step()

    netG.zero_grad()
    y_fake_gens = netD(x_fake)
    loss_G = criterion(y_fake_gens, real_labels)
    loss_G.backward()
    optimG.step()

    if i % 80 == 0:
      print('Epoch {}/{} [{}/{}], loss D real : {:.4f}, loss D fake : {:.4f}, loss G : {:.4f}'.format(
          epoch+1, n_epochs, i, len(dataloader),
          loss_D_real.mean().item(), loss_D_fake.mean().item(),
          loss_G.mean().item()
      ))    
  gens_epochs.append(netG(noise).cpu().detach())

import matplotlib.pyplot as plt
!pip install torch_snippets
from torch_snippets import *

new_noise = torch.randn(20, 100, 1, 1, device=device)
batch_gens = netG(new_noise).cpu().detach()
grid_img = vutils.make_grid((batch_gens), nrow=5)

plt.imshow(grid_img.permute(1, 2, 0))
