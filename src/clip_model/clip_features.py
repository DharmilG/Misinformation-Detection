
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPFeatureExtractor:
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"

        self.processor = CLIPProcessor.from_pretrained(
            self.model_name
        )
        self.model = CLIPModel.from_pretrained(
            self.model_name
        )
        self.model.eval()

    def extract_features(self, image_path, caption):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=[caption],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

        return {
            "image_features": image_features.cpu(),
            "text_features": text_features.cpu()
        }
