import torch
import torch.nn as nn

OUT_DIM = {2: 39, 4: 35, 6: 31}

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class Encoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, latent_dim=50, num_layers=4, num_filters=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers

        convs = [nn.Conv2d(3, num_filters, 3, stride=2), nn.ReLU()]
        for i in range(num_layers-1):
            convs.extend([
                nn.Conv2d(num_filters, num_filters, 3, stride=1),
                nn.ReLU()
            ])
        self.convs = nn.Sequential(*convs)

        out_dim = OUT_DIM[num_layers]
        self.conv_dim = num_filters * out_dim * out_dim
        self.fc = nn.Linear(self.conv_dim, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward_conv(self, obs):
        B = obs.shape[0]
        obs = obs / 255. - 0.5
        obs = self.convs(obs)
        obs = obs.view(B, -1)
        return obs

    def forward(self, obs):
        h = self.forward_conv(obs)
        h = self.fc(h)
        h = self.ln(h)
        h = nonlinearity(h)
        h = self.out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, latent_dim, num_layers=4, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            latent_dim, num_filters * self.out_dim * self.out_dim
        )

        deconvs = []
        for i in range(self.num_layers-1):
            deconvs.extend([
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
                nn.ReLU()
            ])
        deconvs.extend([nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, output_padding=1), nn.ReLU()])
        deconvs.append(nn.Conv2d(num_filters, 3, 3, stride=1, padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, h):
        h = torch.relu(self.fc(h))
        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        deconv = self.deconvs(deconv)
        return deconv
