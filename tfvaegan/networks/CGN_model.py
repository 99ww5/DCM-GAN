import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(opt.nz + opt.attSize, opt.ngh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.ngh, opt.resSize),
        )
        self.apply(weights_init)

    def forward(self, z, c=None):
        if c is None:
            raise ValueError("Generator requires semantic condition c.")

        x = torch.cat((z, c), dim=-1)
        return F.normalize(self.net(x), dim=1)


class Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(opt.resSize + opt.attSize, opt.ndh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.ndh, 1),
        )
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), dim=1)
        return self.net(h)


Discriminator_D1 = Discriminator
