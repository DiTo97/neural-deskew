import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F


class Imagecoder(nn.Module):
    """An image feature extractor with feature pyramid network (FPN)"""

    # TODO: MobileNet v3 in the args
    def __init__(
        self, 
        hidden_dim: int, 
        multiscale: bool,
        dropout: float = 0.25, 
        trainable: bool = False
    ) -> None:
        super().__init__()

        #
        from torchvision.models import mobilenet
        from torchvision.models._api import _get_enum_from_fn as get_enum_from_definition
        from torchvision.models.detection.backbone_utils import mobilenet_backbone as MobileNet_v3
        #
        
        name = "mobilenet_v3_large"
        definition = mobilenet.__dict__[name]

        config = "default".upper()
        checkpoint = get_enum_from_definition(definition)[config]
        
        kwargs = {
            "backbone_name": name, "weights": checkpoint, "fpn": multiscale
        }

        if not trainable:
            kwargs["trainable_layers"] = 0
        
        self.model = MobileNet_v3(**kwargs)
        self.multiscale = multiscale

        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(256, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)

        if self.multiscale:
            vals = [val for key, val in x.items() if key != "pool"]

            x, vals = vals[0], vals[1:]

            # It fuses FPN maps
            for val in vals:
                x *= val

        x = self.avgpool2d(x)
        x = x.squeeze((2, 3))
        x = self.dropout(x)
        
        x = self.project(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class Anglecoder(nn.Module):
    """An angle feature extractor"""

    def __init__(self, hidden_dim: int) -> None:
        pass


class Encoder(nn.Module):
    """A base encoder model for angle confidence vectors"""

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super(Encoder, self).__init__()

        self.conv1d = nn.Conv1d(
            num_channels, hidden_dim, kernel_size=kernel_size, stride=1, padding=1
        )

        self.batch_norm1d = nn.BatchNorm1d(hidden_dim)

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=3)
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x = self.batch_norm1d(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x
