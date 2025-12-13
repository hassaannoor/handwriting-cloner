import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

transform = T.Compose([
    T.Grayscale(),
    T.Resize((128, 512)),
    T.ToTensor()
])

class StyleDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.writers = sorted(os.listdir(root))

        for idx, writer in enumerate(self.writers):
            wdir = os.path.join(root, writer)
            for img in os.listdir(wdir):
                self.samples.append(
                    (os.path.join(wdir, img), idx)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return transform(img), label
