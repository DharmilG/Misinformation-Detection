
import numpy as np
from pathlib import Path
from clip_features import CLIPFeatureExtractor

# Load the sample image
image_path = "data/sample/test_3.webp"

# Create CLIP extractor
extractor = CLIPFeatureExtractor()

# Extract image and text features
features = extractor.extract_features(
    image_path=image_path,
    caption="A sample image for misinformation detection"
)

image_features = features["image_features"]
text_features = features["text_features"]

# Verify extracted features
assert image_features.shape == (1, 512)
assert text_features.shape == (1, 512)

assert np.isfinite(image_features.numpy()).all()
assert np.isfinite(text_features.numpy()).all()

print("Image features shape:", image_features.shape)
print("Text features shape:", text_features.shape)
print("CLIP loading and feature extraction verified!")
