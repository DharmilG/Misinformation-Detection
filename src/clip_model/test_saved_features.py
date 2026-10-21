
import numpy as np
from pathlib import Path

# Load features extracted from the third image
feature_path = Path("data/sample/clip_features_3.npz")

data = np.load(feature_path)

image_features = data["image_features"]
text_features = data["text_features"]

print("Third image features loaded successfully!")
print("Image features shape:", image_features.shape)
print("Text features shape:", text_features.shape)

assert image_features.shape == (1, 512)
assert text_features.shape == (1, 512)

assert np.isfinite(image_features).all()
assert np.isfinite(text_features).all()

print("Third image features loaded successfully!")
