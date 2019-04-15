import argparse
import os
import itertools
import random
import numpy as np
import torch
import torch.cuda
import torch.optim
import torch.autograd as autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutil

os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
  # folder dataset
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.CenterCrop(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
  nc = 3
elif opt.dataset == 'lsun':
  dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                      transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.CenterCrop(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
  nc = 3
elif opt.dataset == 'cifar10':
  dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                           transforms.Resize(opt.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))
  nc = 3
elif opt.dataset == 'mnist':
  dataset = dset.MNIST(root=opt.dataroot, download=True,
                       transform=transforms.Compose([
                         transforms.Resize(opt.img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,)),
                       ]))
  nc = 1
elif opt.dataset == 'fake':
  dataset = dset.FakeData(image_size=(3, opt.img_size, opt.img_size),
                          transform=transforms.ToTensor())
  nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

nz = int(opt.latent_dim)


def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm") != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
  """Returns one-hot encoded Variable"""
  y_cat = np.zeros((y.shape[0], num_columns))
  y_cat[range(y.shape[0]), y] = 1.0

  return autograd.Variable(FloatTensor(y_cat))


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    input_dim = nz + opt.n_classes + opt.code_dim

    self.init_size = opt.img_size // 4  # Initial size before upsampling
    self.l1 = nn.Sequential(
      nn.Linear(input_dim, 128 * self.init_size ** 2))

    self.conv_blocks = nn.Sequential(
      nn.BatchNorm2d(128),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, 3, stride=1, padding=1),
      nn.BatchNorm2d(128, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, 3, stride=1, padding=1),
      nn.BatchNorm2d(64, 0.8),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, nc, 3, stride=1, padding=1),
      nn.Tanh(),
    )

  def forward(self, noise, label, code):
    gen_input = torch.cat((noise, label, code), -1)
    out = self.l1(gen_input)
    out = out.view(out.shape[0], 128, self.init_size, self.init_size)
    img = self.conv_blocks(out)
    return img


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    def discriminator_block(in_filters, out_filters, bn=True):
      """Returns layers of each discriminator block"""
      block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
               nn.LeakyReLU(0.2, inplace=True),
               nn.Dropout2d(0.25)]
      if bn:
        block.append(nn.BatchNorm2d(out_filters, 0.8))
      return block

    self.conv_blocks = nn.Sequential(
      *discriminator_block(nc, 64, bn=False),
      *discriminator_block(64, 64),
      *discriminator_block(64, 128),
      *discriminator_block(128, 128),
    )

    # The height and width of downsampled image
    ds_size = opt.img_size // 2 ** 4

    # Output layers
    self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
    self.aux_layer = nn.Sequential(
      nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
    self.latent_layer = nn.Sequential(
      nn.Linear(128 * ds_size ** 2, opt.code_dim))

  def forward(self, img):
    out = self.conv_blocks(img)
    out = out.view(out.shape[0], -1)
    validity = self.adv_layer(out)
    label = self.aux_layer(out)
    latent_code = self.latent_layer(out)

    return validity, label, latent_code


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
netG = Generator()
netD = Discriminator()

if cuda:
  netG.cuda()
  netD.cuda()
  adversarial_loss.cuda()
  categorical_loss.cuda()
  continuous_loss.cuda()

# Initialize weights
netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

# Optimizers
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerInfo = torch.optim.Adam(
  itertools.chain(netG.parameters(), netD.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = autograd.Variable(torch.zeros(opt.n_classes ** 2, nz, device=device))
static_label = to_categorical(
  np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = autograd.Variable(torch.zeros(opt.n_classes ** 2, opt.code_dim, device=device))


def sample_image(n_row, batches_done):
  """Saves a grid of generated digits ranging from 0 to n_classes"""
  # Static sample
  z = torch.randn(n_row ** 2, nz, device=device)
  static_sample = netG(z, static_label, static_code)
  vutil.save_image(static_sample.data,
                   f"images/static/{batches_done}.png", nrow=n_row, normalize=True)

  # Get varied c1 and c2
  zeros = np.zeros((n_row ** 2, 1))
  c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
  c1 = autograd.Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
  c2 = autograd.Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
  sample1 = netG(static_z, static_label, c1)
  sample2 = netG(static_z, static_label, c2)
  vutil.save_image(
    sample1.data, f"images/varying_c1/{batches_done}.png", nrow=n_row, normalize=True)
  vutil.save_image(
    sample2.data, f"images/varying_c2/{batches_done}.png", nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
  for i, (real_imgs, labels) in enumerate(dataloader):

    batch_size = real_imgs.shape[0]

    # Adversarial ground truths
    valid = autograd.Variable(FloatTensor(
      batch_size, 1).fill_(1.0), requires_grad=False)
    fake = autograd.Variable(FloatTensor(
      batch_size, 1).fill_(0.0), requires_grad=False)

    # Configure input
    real_imgs = autograd.Variable(real_imgs.type(FloatTensor))
    labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

    # -----------------
    #  Train Generator
    # -----------------

    optimizerG.zero_grad()

    # Sample noise and labels as generator input
    z = autograd.Variable(FloatTensor(
      np.random.normal(0, 1, (batch_size, nz))))
    label_input = to_categorical(np.random.randint(
      0, opt.n_classes, batch_size), num_columns=opt.n_classes)
    code_input = autograd.Variable(FloatTensor(
      np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

    # Generate a batch of images
    gen_imgs = netG(z, label_input, code_input)

    # Loss measures generator's ability to fool the discriminator
    validity, _, _ = netD(gen_imgs)
    errG = adversarial_loss(validity, valid)

    errG.backward()
    optimizerG.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizerD.zero_grad()

    # Loss for real images
    real_pred, _, _ = netD(real_imgs)
    d_real_loss = adversarial_loss(real_pred, valid)

    # Loss for fake images
    fake_pred, _, _ = netD(gen_imgs.detach())
    d_fake_loss = adversarial_loss(fake_pred, fake)

    # Total discriminator loss
    errD = (d_real_loss + d_fake_loss) / 2

    errD.backward()
    optimizerD.step()

    # ------------------
    # Information Loss
    # ------------------

    optimizerInfo.zero_grad()

    # Sample labels
    sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

    # Ground truth labels
    gt_labels = autograd.Variable(LongTensor(
      sampled_labels), requires_grad=False)

    # Sample noise, labels and code as generator input
    z = autograd.Variable(FloatTensor(
      np.random.normal(0, 1, (batch_size, nz))))
    label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
    code_input = autograd.Variable(FloatTensor(
      np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

    gen_imgs = netG(z, label_input, code_input)
    _, pred_label, pred_code = netD(gen_imgs)

    errInfo = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
      pred_code, code_input
    )

    errInfo.backward()
    optimizerInfo.step()

    # --------------
    # Log Progress
    # --------------

    print(f"[Epoch {epoch}/{opt.n_epochs}] "
          f"[Batch {i}/{len(dataloader)}] "
          f"[D loss: {errD.item():.4f}] "
          f"[G loss: {errG.item():.4f}] "
          f"[info loss: {errInfo.item():.4f}]")

    batches_done = epoch * len(dataloader) + i
    if batches_done % 500 == 0:
      sample_image(n_row=10, batches_done=batches_done)
