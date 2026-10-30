"""
src/utils/config.py
===================
Application-wide constants.

All pipeline modules import from here — never from each other — so that
changing a limit or version string is a single-file edit.
"""
from __future__ import annotations

import pathlib

# ── Image validation ──────────────────────────────────────────────────────
MAX_IMAGE_BYTES: int        = 10 * 1024 * 1024   # 10 MB
SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = ("JPEG", "PNG", "WEBP", "GIF", "BMP")
SUPPORTED_EXTENSIONS: tuple[str, ...]    = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
MAX_LONG_EDGE_PX: int       = 1024    # images larger than this are downscaled
CLIP_MAX_TOKENS: int        = 77      # hard limit in CLIP tokenizer

# ── Caption validation ────────────────────────────────────────────────────
MIN_CAPTION_CHARS: int      = 3
MAX_CAPTION_CHARS: int      = 2000

# ── Retrieval ─────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int        = 5

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CORPUS_PATH  = DATA_DIR / "corpus.json"
SAMPLES_PATH = DATA_DIR / "samples" / "samples.json"

# ── Version strings (increment when the artefact changes) ────────────────
MODEL_VERSION            = "open-clip-ViT-B-32-laion2b-s34b-b79k-v1"
ENCODER_REVISION         = "laion2b_s34b_b79k"
FEATURE_CONTRACT_VERSION = "v1"
CORPUS_VERSION           = "synthetic-v1"
CORPUS_MANIFEST_SHA256   = "0000000000000000"   # updated by ingest script
RETRIEVAL_CONFIG_HASH    = "tfidf-cosine-v1"
EMBEDDING_REVISION       = "none-tfidf"
CHUNKING_VERSION         = "sentence-v1"
PROMPT_VERSION           = "keyword-heuristic-v1"

# ── Classifier ────────────────────────────────────────────────────────────
CLASSIFIER_THRESHOLD: float = 0.5   # score >= threshold → MATCHED
