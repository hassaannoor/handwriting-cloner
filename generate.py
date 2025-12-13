import torch
from models.model import TRGAN

class HandwritingGenerator:
    def __init__(self, model_path):
        self.model = TRGAN()
        self.model.netG.load_state_dict(torch.load(model_path))
        self.model.eval()

    def generate(self, text_encoded, style_image):
        # style_image would be injected here in a real implementation
        return self.model.netG(text_encoded)
