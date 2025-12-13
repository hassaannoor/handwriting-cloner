import torch
from models.model import TRGAN
from generator_dataset import GeneratorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = GeneratorDataset("train_pairs.pickle")
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = TRGAN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(50):
    for img, text, style in loader:
        img = img.to(DEVICE)
        style = style.to(DEVICE)

        pred = model(text, style)
        loss = ((pred - img) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} complete")

torch.save(model.state_dict(), "handwriting_generator.pth")
