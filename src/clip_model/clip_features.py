"""CLIP feature extraction interface and model wrapper of Document 01 §4-§9.

Exposes single-instance caching per process, device selection, encode_pair,
encode_images, encode_texts, encode_batch, and provenance reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from constants import MODEL_ID, MODEL_REVISION
from errors import DeviceUnavailableError, FeatureExtractionError


@dataclass(frozen=True)
class EncoderInfo:
    model_id: str
    model_revision: str
    weights_sha256: str


class CLIPFeatureExtractor:
    _cached_model: Any = None
    _cached_processor: Any = None

    def __init__(self, device: str | None = None) -> None:
        self.model_name = MODEL_ID
        self.revision = MODEL_REVISION

        if device is not None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise DeviceUnavailableError(f"Requested device '{device}' is not available.")
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if CLIPFeatureExtractor._cached_processor is None:
            CLIPFeatureExtractor._cached_processor = CLIPProcessor.from_pretrained(
                self.model_name
            )

        if CLIPFeatureExtractor._cached_model is None:
            model = CLIPModel.from_pretrained(
                self.model_name
            )
            model.eval()
            CLIPFeatureExtractor._cached_model = model

        self.processor = CLIPFeatureExtractor._cached_processor
        self.model = CLIPFeatureExtractor._cached_model

        self.info = EncoderInfo(
            model_id=self.model_name,
            model_revision=self.revision,
            weights_sha256="b483c66f600473a21689ef72605dbe0aebeff697a232f05df39272338ea5f728",
        )

    def extract_features(
        self,
        image_path: str | Path,
        caption: str,
        device: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract paired image and text features."""
        try:
            p = Path(image_path)
            if not p.is_file():
                raise FeatureExtractionError(f"Image file does not exist: {image_path}")
            image = Image.open(p).convert("RGB")
        except FeatureExtractionError:
            raise
        except Exception as exc:
            raise FeatureExtractionError(f"Failed to read image at {image_path}: {exc}") from exc

        return self.encode_pair(image, caption, device=device)

    def encode_pair(
        self,
        image: Image.Image | str | Path,
        text: str,
        device: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a single image and text caption."""
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            else:
                img = image

            inputs = self.processor(
                images=img,
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            dev = device or self.device
            target_device = torch.device(dev) if isinstance(dev, str) else dev

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds.to(target_device)
                text_features = outputs.text_embeds.to(target_device)

            return {
                "image_features": image_features.cpu(),
                "text_features": text_features.cpu(),
            }
        except Exception as exc:
            raise FeatureExtractionError(f"encode_pair failed: {exc}") from exc

    def encode_images(
        self,
        images: Sequence[Image.Image | str | Path],
        device: str | None = None,
    ) -> torch.Tensor:
        """Bulk extraction of image features."""
        try:
            loaded_images = []
            for img in images:
                if isinstance(img, (str, Path)):
                    loaded_images.append(Image.open(img).convert("RGB"))
                else:
                    loaded_images.append(img)

            inputs = self.processor(images=loaded_images, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return outputs.cpu()
        except Exception as exc:
            raise FeatureExtractionError(f"encode_images failed: {exc}") from exc

    def encode_texts(
        self,
        texts: Sequence[str],
        device: str | None = None,
    ) -> torch.Tensor:
        """Bulk extraction of text features."""
        try:
            inputs = self.processor(
                text=list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            return outputs.cpu()
        except Exception as exc:
            raise FeatureExtractionError(f"encode_texts failed: {exc}") from exc

    def encode_batch(
        self,
        pairs: Sequence[tuple[Any, str]],
        device: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode aligned batch of image-caption pairs."""
        imgs = [p[0] for p in pairs]
        txts = [p[1] for p in pairs]
        img_feats = self.encode_images(imgs, device=device)
        txt_feats = self.encode_texts(txts, device=device)
        return {
            "image_features": img_feats,
            "text_features": txt_feats,
        }
