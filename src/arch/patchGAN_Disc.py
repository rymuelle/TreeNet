from torch import nn
#https://github.com/ramanakumars/patchGAN/blob/main/patchgan/disc.py


from torch.nn.parameter import Parameter


class Transferable():
    def __init__(self):
        super(Transferable, self).__init__()

    def load_transfer_data(self, state_dict):
        own_state = self.state_dict()
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape == own_state[name].data.shape:
                own_state[name].copy_(param)
                count += 1

        if count > 0:
            print(f"Loaded {count} weights out of {len(state_dict)}")
        else:
            raise InvalidCheckpointError("Could not load transfer weights")


class InvalidCheckpointError(Exception):
    pass


class Discriminator(nn.Module, Transferable):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm=False, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.Tanh(),
            ]
            if norm:
                sequence += [norm_layer(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.Tanh(),
        ]
        if norm:
            sequence += [norm_layer(ndf * nf_mult)]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw,
                               stride=1, padding=padw), nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)