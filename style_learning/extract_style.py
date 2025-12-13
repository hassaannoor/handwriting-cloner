import torch
from PIL import Image
from style_encoder import StyleEncoder, transform
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = StyleEncoder().to(DEVICE)
model.load_state_dict(torch.load("style_encoder.pth"))
model.eval()

imgs = []
path = "style_dataset/writer_me"

for f in os.listdir(path):
    img = Image.open(os.path.join(path, f)).convert("L")
    imgs.append(transform(img))

x = torch.stack(imgs).to(DEVICE)

with torch.no_grad():
    style_vector = model(x).mean(dim=0)

torch.save(style_vector, "my_style.pt")
print("Saved style vector as my_style.pt")
