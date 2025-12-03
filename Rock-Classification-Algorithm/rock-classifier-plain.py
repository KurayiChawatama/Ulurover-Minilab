import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models import efficientnet_b2


class RockClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.base = efficientnet_b2(weights=None)

        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 7)
        )

        self.attention_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.base.features(x)
        x = self.attention_pool(x)
        x = x.view(x.size(0), -1)
        return self.base.classifier(x)


model = RockClassifier()

model.load_state_dict(torch.load("./rock_classifier_efficientnet_b0.pth", map_location=torch.device('cpu')))

model.eval()

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = "../Images/Test-Rocks/limestone.jpeg"

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

print(torch.argmax(output, dim=1).item())
