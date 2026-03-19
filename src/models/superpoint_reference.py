import torch


class SuperPointReference(torch.nn.Module):  # noqa: WPS230 (Found too many public instance attributes)
    """Pytorch definition of SuperPoint Network.
    Reference Implementation from
    https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
    """

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        c1 = 64
        c2 = 64
        c3 = 128
        c4 = 128
        c5 = 256
        d1 = 256
        d2 = 65

        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, d2, kernel_size=1, stride=1, padding=0)

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, batch):
        # Shared Encoder.
        x1 = self.relu(self.conv1a(batch))
        x2 = self.relu(self.conv1b(x1))
        x3 = self.pool(x2)
        x4 = self.relu(self.conv2a(x3))
        x5 = self.relu(self.conv2b(x4))
        x6 = self.pool(x5)
        x7 = self.relu(self.conv3a(x6))
        x8 = self.relu(self.conv3b(x7))
        x9 = self.pool(x8)
        x10 = self.relu(self.conv4a(x9))
        x11 = self.relu(self.conv4b(x10))

        # Detector Head.
        cPa = self.relu(self.convPa(x11))
        semi = self.convPb(cPa)

        # Descriptor Head.
        cDa = self.relu(self.convDa(x11))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))

        return semi, desc
