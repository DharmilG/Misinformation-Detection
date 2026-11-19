"""Add four CLIP / D-C explainer slides to the existing presentation in place.

This script NEVER regenerates the deck. It opens the file as it is, appends four
slides in the same visual style, then moves them into position after the
"Architecture explained" slide. All existing slides and any manual edits are
left untouched.

Idempotent: if the explainer slides are already present it does nothing.

Usage:
    python scripts/add_clip_slides.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import build_presentation as bp
from build_presentation import (  # noqa: E402
    BLACK, CREAM, GREY, WHITE, LIGHT, add_fade, base_slide,
    card, footer, para, rule, textbox,
)

ROOT = Path(__file__).resolve().parent.parent
MARKER = "WHAT IS CLIP?"


def slide_clip_basics(prs):
    s = base_slide(prs, "The encoder", "What is CLIP?")
    card(s, 0.9, 2.35, 11.4, 1.5)
    tf = textbox(s, 1.25, 2.6, 10.7, 1.1)
    para(tf, "CLIP = Contrastive Language–Image Pretraining.", size=26, bold=True,
         first=True, space_after=8)
    para(tf, "It was trained to match images with the text that describes them.",
         size=20, color=GREY)

    items = [
        "It learns one shared space for pictures and words.",
        "Things that mean the same thing land close together in that space.",
        "We use a frozen, pinned CLIP ViT-B/32 checkpoint — it never changes.",
        "Because of that, every score we produce is comparable across runs.",
    ]
    t2 = textbox(s, 0.9, 4.2, 11.4, 2.6)
    for i, item in enumerate(items):
        para(t2, f"{i+1}.  {item}", size=20, space_after=14, first=(i == 0))
    footer(s, "Frozen means: we use it as a fixed measuring tool, not something we retrain.")


def slide_clip_architecture(prs):
    s = base_slide(prs, "The encoder", "How CLIP sees the world",
                   subtitle="Two encoders. One shared space.")

    def box(x, y, w, h, label, *, fill=WHITE, size=15):
        shape = card(s, x, y, w, h, fill=fill)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.06)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(tf, label, size=size, bold=True, align=PP_ALIGN.CENTER, first=True,
             color=WHITE if fill == BLACK else BLACK)
        return shape

    def arrow(x1, y1, x2, y2):
        line = s.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
        )
        line.line.color.rgb = BLACK
        line.line.width = Pt(1.5)
        line.line.end_arrowhead = True

    # image path
    box(0.9, 2.65, 2.0, 1.1, "IMAGE\nphoto", size=14)
    box(3.4, 2.65, 2.6, 1.1, "IMAGE ENCODER\nViT → 512 numbers", size=13)
    arrow(2.9, 3.2, 3.4, 3.2)

    # text path
    box(0.9, 4.15, 2.0, 1.1, "TEXT\ncaption", size=14)
    box(3.4, 4.15, 2.6, 1.1, "TEXT ENCODER\n→ 512 numbers", size=13)
    arrow(2.9, 4.7, 3.4, 4.7)

    # shared space
    box(7.0, 3.1, 2.6, 1.8, "SHARED SPACE\nimage vector\ntext vector", fill=BLACK, size=14)
    arrow(6.0, 3.2, 7.0, 3.6)
    arrow(6.0, 4.7, 7.0, 4.4)

    box(10.3, 3.35, 2.1, 1.3, "COSINE\nSIMILARITY", size=14)
    arrow(9.6, 4.0, 10.3, 4.0)

    tf = textbox(s, 0.9, 5.85, 11.4, 0.8)
    para(tf, "One space. Two inputs. A single number says how close they are.",
         size=19, bold=True, first=True)
    footer(s, "We keep the encoders fixed and only read the vectors they produce.")


def slide_why_embeddings(prs):
    s = base_slide(prs, "The encoder", "Why only the embeddings?")
    items = [
        "They are small: 512 numbers per image or caption.",
        "They are stable: the same input gives the same vector every time.",
        "They are reusable: cache once, score thousands of pairs cheaply.",
        "They are fair: a fixed encoder means we compare like with like.",
        "We do not need CLIP to make the decision — only to describe both sides.",
    ]
    t = textbox(s, 0.9, 2.35, 11.4, 3.6)
    for i, item in enumerate(items):
        para(t, f"{i+1}.  {item}", size=20, space_after=16, first=(i == 0))
    card(s, 0.9, 5.75, 11.4, 0.95, fill=BLACK, line_color=BLACK)
    tf = textbox(s, 1.25, 5.95, 10.7, 0.7)
    para(tf, "CLIP measures. The D-C head decides.", size=22, bold=True,
         color=WHITE, first=True)


def slide_dc_head(prs):
    s = base_slide(prs, "The classifier", "What is the D-C head?")
    card(s, 0.9, 2.35, 11.4, 1.3)
    tf = textbox(s, 1.25, 2.6, 10.7, 0.9)
    para(tf, "A small classifier trained on our data. D = the features we build. "
             "C = the candidate family.", size=19, bold=True, first=True)

    steps = [
        ("1", "Build features", "image · text · image×text · |image−text| → 2,048 numbers"),
        ("2", "Small neural net", "2,048 → 256 → 2 scores"),
        ("3", "Decision", "MATCHED or MISMATCHED at threshold 0.4507"),
    ]
    for i, (num, title, detail) in enumerate(steps):
        y = 3.95 + i * 0.9
        card(s, 0.9, y, 11.4, 0.8)
        t1 = textbox(s, 1.1, y + 0.16, 0.6, 0.5)
        para(t1, num, size=20, bold=True, color=GREY, first=True)
        t2 = textbox(s, 1.7, y + 0.15, 3.0, 0.5)
        para(t2, title, size=19, bold=True, first=True)
        t3 = textbox(s, 4.9, y + 0.17, 7.1, 0.5)
        para(t3, detail, size=16, color=GREY, first=True)

    footer(s, "Calibration temperature 1.285 turns the score into a readable probability.")


NEW_SLIDES = [slide_clip_basics, slide_clip_architecture, slide_why_embeddings, slide_dc_head]


def add_to(path: Path) -> tuple[int, str]:
    prs = Presentation(str(path))
    existing = "\n".join(
        sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame
    )
    if MARKER in existing:
        return 0, "explainer slides already present - no changes made"

    # Append four slides using the existing deck's theme and helpers.
    for fn in NEW_SLIDES:
        fn(prs)

    # Move the four new slides from the end to just after slide 6.
    ids = list(prs.slides._sldIdLst)
    moved = ids[-len(NEW_SLIDES):]
    for element in moved:
        prs.slides._sldIdLst.remove(element)
    for offset, element in enumerate(moved):
        prs.slides._sldIdLst.insert(6 + offset, element)

    prs.save(str(path))
    return len(moved), f"inserted {len(moved)} slides after position 6 in {path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=ROOT / "Project_Presentation.pptx")
    args = parser.parse_args()
    count, message = add_to(args.deck)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
