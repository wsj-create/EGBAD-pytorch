#导入相关包
import torch
from torch import nn



"""定义生成器网络结构"""
class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.ReLU(inplace=True), bn=True):
        seq = []
        seq += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn is True:
          seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(20, 64*8, stride=1, padding=0)]
    seq += [CBA(64*8, 64*4)]
    seq += [CBA(64*4, 64*2)]
    seq += [CBA(64*2, 64)]
    seq += [CBA(64, 1, activation=nn.Tanh(), bn=False)]

    self.generator_network = nn.Sequential(*seq)

  def forward(self, z):
      out = self.generator_network(z)

      return out


"""定义判别器网络结构"""
class Discriminator(nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.LeakyReLU(0.1, inplace=True)):
        seq = []
        seq += [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(1, 64)]
    seq += [CBA(64, 64*2)]
    seq += [CBA(64*2, 64*4)]
    seq += [CBA(64*4, 64*8)]
    seq += [nn.Conv2d(64*8, 512, kernel_size=4, stride=1)]
    self.feature_network = nn.Sequential(*seq)

    seq = []
    seq += [nn.Linear(20, 512)]
    seq += [nn.BatchNorm1d(512)]
    seq += [nn.LeakyReLU(0.1, inplace=True)]
    self.latent_network = nn.Sequential(*seq)

    self.critic_network = nn.Linear(1024, 1)

  def forward(self, x, z):
      feature = self.feature_network(x)
      feature = feature.view(feature.size(0), -1)
      latent = self.latent_network(z)

      out = self.critic_network(torch.cat([feature, latent], dim=1))

      return out, feature


"""定义编码器结构"""
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.LeakyReLU(0.1, inplace=True)):
        seq = []
        seq += [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(1, 64)]
    seq += [CBA(64, 64*2)]
    seq += [CBA(64*2, 64*4)]
    seq += [CBA(64*4, 64*8)]
    seq += [nn.Conv2d(64*8, 512, kernel_size=4, stride=1)]
    self.feature_network = nn.Sequential(*seq)

    self.embedding_network = nn.Linear(512, 20)

  def forward(self, x):
    feature = self.feature_network(x).view(-1, 512)
    z = self.embedding_network(feature)

    return z


if __name__ == '__main__':
    x = torch.ones((2, 1, 64, 64))
    z = torch.ones((2, 20, 1, 1))
    Generator = Generator()
    Discriminator = Discriminator()
    Encoder = Encoder()
    output_G = Generator(z)
    output_D1, output_D2= Discriminator(x, z.view(2, -1))
    output_E = Encoder(x)
    print(output_G.shape)
    print(output_D1.shape)
    print(output_D2.shape)
    print(output_E.shape)



