import torch.nn as nn

def fc_block(in_dim, out_dim, activation=True, dropout=False):
    modules = [
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim)
    ]

    if dropout:
        modules.append(nn.Dropout())

    if activation:
        modules += [
            nn.ReLU()
    ]

    return nn.Sequential(*modules)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_transpose_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2,  padding=2, output_padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
