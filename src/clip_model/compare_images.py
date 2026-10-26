import numpy as np

data1 = np.load("data/sample/clip_features.npz")
data2 = np.load("data/sample/clip_features_3.npz")

image1 = data1["image_features"].flatten()
image2 = data2["image_features"].flatten()

similarity = np.dot(image1, image2) / (
    np.linalg.norm(image1) * np.linalg.norm(image2)
)

print("Image 1 shape:", image1.shape)
print("Image 2 shape:", image2.shape)
print("Features exactly identical:", np.array_equal(image1, image2))
print("Cosine similarity:", round(float(similarity), 6))