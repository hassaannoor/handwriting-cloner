import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from style_dataset import StyleDataset
from style_encoder import StyleEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = StyleDataset("style_dataset")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = StyleEncoder().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss {total_loss:.4f}")

torch.save(model.state_dict(), "style_encoder.pth")
