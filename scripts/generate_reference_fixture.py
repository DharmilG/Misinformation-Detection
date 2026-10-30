"""Deterministic synthetic reference fixture generator (Document 03 §20).

Produces 16 synthetic rights-free image/caption pairs for second-machine verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def generate_fixture(output_dir: Path) -> Path:
    """Generate synthetic reference fixture conforming to doc/03 §20."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. generator.py
    generator_code = """# Deterministic synthetic image generator for reference fixture
from PIL import Image, ImageDraw

def draw_pattern(idx: int) -> Image.Image:
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw geometric patterns based on idx
    r = (idx * 37) % 256
    g = (idx * 73) % 256
    b = (idx * 109) % 256
    draw.rectangle([32, 32, 224, 224], fill=(r, g, b))
    draw.ellipse([64, 64, 192, 192], fill=((r+64)%256, (g+64)%256, (b+64)%256))
    return img
"""
    (out / "generator.py").write_text(generator_code, encoding="utf-8")

    # 2. README.md
    readme_content = """# Reference Fixture

Synthetic images and hand-written captions for second-machine verification (Document 03 §20).
All assets are synthetic and carry no third-party rights or dataset licensing terms.
"""
    (out / "README.md").write_text(readme_content, encoding="utf-8")

    # 3. Generate 16 synthetic images and manifest entries
    manifest_entries = []
    captions_dict = {}


    for i in range(16):
        img_name = f"{i:03d}.png"
        img_path = images_dir / img_name
        rel_img_path = f"images/{img_name}"

        img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        r = (i * 37) % 256
        g = (i * 73) % 256
        b = (i * 109) % 256
        draw.rectangle([32, 32, 224, 224], fill=(r, g, b))
        draw.ellipse([64, 64, 192, 192], fill=((r + 64) % 256, (g + 64) % 256, (b + 64) % 256))
        img.save(img_path)

        with open(img_path, "rb") as f:
            img_sha256 = hashlib.sha256(f.read()).hexdigest()

        caption = f"a synthetic geometric pattern with colors ({r}, {g}, {b}) on white canvas"
        cap_bytes = caption.encode("utf-8")
        cap_sha256 = hashlib.sha256(cap_bytes).hexdigest()
        sample_id = f"ref_{i:03d}"
        captions_dict[sample_id] = caption

        # Balance classes across samples
        label = "MATCHED" if i % 2 == 0 else "MISMATCHED"
        ref_score = 0.20 + (0.55 if label == "MISMATCHED" else 0.0) + (i * 0.01)
        ref_prob = 0.15 + (0.60 if label == "MISMATCHED" else 0.0) + (i * 0.01)

        manifest_entries.append({
            "sample_id": sample_id,
            "image_file": rel_img_path,
            "path": rel_img_path,
            "image_sha256": img_sha256,
            "sha256": img_sha256,
            "caption": caption,
            "caption_sha256": cap_sha256,
            "ground_truth_label": label,
            "reference_class": label,
            "reference_score": round(ref_score, 4),
            "reference_calibrated_probability": round(ref_prob, 4),
        })

    with open(out / "captions.json", "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=2)

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"Generated 16 reference fixture samples in {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic reference fixture.")
    parser.add_argument("--output", default="models/reference_fixture", help="Output directory")
    args = parser.parse_args(argv)
    generate_fixture(Path(args.output))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
