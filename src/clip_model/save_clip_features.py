
import torch
import numpy as np
from pathlib import Path
from clip_features import CLIPFeatureExtractor

# Project paths
image_path = "data/sample/test_3.webp"
output_path = Path("data/sample/clip_features_3.npz")
# Create the extractor
extractor = CLIPFeatureExtractor()

# Extract image and text features
features = extractor.extract_features(
    image_path=image_path,
    caption="A sample image for misinformation detection"
)

# Convert tensors to NumPy arrays
image_features = features["image_features"].numpy()
text_features = features["text_features"].numpy()

# Save features
np.savez(
    output_path,
    image_features=image_features,
    text_features=text_features
)

print("CLIP features saved successfully!")
print("Saved to:", output_path)
print("Image features shape:", image_features.shape)
print("Text features shape:", text_features.shape)
