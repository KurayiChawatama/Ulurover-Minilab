import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b2
from PIL import Image
import io
from shiny import App, ui, render

# Define Rock Classifier Model
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

# Load trained model
model = RockClassifier()
model.load_state_dict(torch.load("./rock_classifier_efficientnet_b0.pth", map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Define transform for images
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Rock categories (adjust as needed)
rock_classes = ["Basalt", "Coal", "Granite", "Limestone", "Marble", "Quartzite", "Sandstone"]

# Define UI
app_ui = ui.page_fluid(
    ui.h2("Rock Classifier"),
    ui.p("Upload an image of a rock to classify it."),
    ui.input_file("file1", "Upload Image", multiple=False, accept=["image/jpeg", "image/png"]),
    ui.output_image("uploaded_img"),
    ui.output_text("prediction")
)

# Define Server logic
def server(input, output, session):
    @render.text
    def prediction():
        file = input.file1()
        if file and len(file) > 0:  # Ensure a file is uploaded
            image_data = file[0]["datapath"]  # FIX: Use file[0] to access the first file
            image = Image.open(image_data).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)

            predicted_class = rock_classes[torch.argmax(output, dim=1).item()]
            return f"Predicted Rock Type: {predicted_class}"

        return "Upload an image to classify."

    @render.image
    def uploaded_img():
        file = input.file1()
        if file:
            return {"src": file[0]["datapath"], "height": "300px"}  # Fix: Use file[0] to access first file
        return None


# Run the app
app = App(app_ui, server)
app.run()

