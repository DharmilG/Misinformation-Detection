
from pathlib import Path
import sys

# Make the CLIP module accessible
PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_DIR = PROJECT_ROOT / "src" / "clip_model"

sys.path.insert(0, str(CLIP_DIR))

from clip_features import CLIPFeatureExtractor


def main():
    image_path = PROJECT_ROOT / "data" / "sample" / "test_3.webp"

    if not image_path.exists():
        print("Error: Sample image not found:", image_path)
        return

    print("Loading CLIP model and extracting features...")

    extractor = CLIPFeatureExtractor()

    features = extractor.extract_features(
        image_path=str(image_path),
        caption="A sample image for misinformation detection"
    )

    image_features = features["image_features"]
    text_features = features["text_features"]

    print("Image features shape:", tuple(image_features.shape))
    print("Text features shape:", tuple(text_features.shape))
    print("CLIP integration successful!")


if __name__ == "__main__":
    main()
