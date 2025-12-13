import torch
import torch.nn as nn
import torchvision.transforms as T

class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.hub.load(
            'pytorch/vision', 'resnet18', pretrained=True
        )
        self.encoder.fc = nn.Linear(512, 256)

    def forward(self, x):
        return self.encoder(x)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((128,512))
])
