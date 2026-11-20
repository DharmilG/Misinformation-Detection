"""Add four data / preprocessing / model-selection slides to the deck in place.

Opens Project_Presentation.pptx as it is, appends four slides in the existing
style, inserts them into the data and classifier sections, and fixes one stale
conformance figure. Existing slides and manual edits are otherwise untouched.

Idempotent: does nothing if the new slides are already present.

Usage:
    python scripts/add_data_slides.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR

from build_presentation import (
    BLACK, CREAM, GREY, WHITE, LIGHT, base_slide, card, footer, para, textbox,
)

ROOT = Path(__file__).resolve().parent.parent
MARKER = "Real pairs. Plausible fakes."


def slide_real_vs_fakes(prs):
    s = base_slide(prs, "The data", "Real pairs. Plausible fakes.")
    card(s, 0.9, 2.35, 5.55, 2.5)
    tf = textbox(s, 1.25, 2.6, 4.9, 2.1)
    para(tf, "CLASS 0 — AUTHENTIC", size=14, bold=True, color=GREY, first=True, space_after=10)
    para(tf, "A real photograph with its verified news caption.", size=21, bold=True, space_after=10)
    para(tf, "They genuinely belong together.", size=17, color=GREY)

    card(s, 6.75, 2.35, 5.55, 2.5)
    tf = textbox(s, 7.1, 2.6, 4.9, 2.1)
    para(tf, "CLASS 1 — HARD SWAP", size=14, bold=True, color=GREY, first=True, space_after=10)
    para(tf, "A real caption deliberately moved to a different real photo.",
         size=21, bold=True, space_after=10)
    para(tf, "The swap is chosen to look believable.", size=17, color=GREY)

    tf = textbox(s, 0.9, 5.2, 11.4, 1.5)
    para(tf, "How the hard swaps are chosen", size=17, bold=True, color=GREY,
         first=True, space_after=10)
    para(tf, "CLIP text similarity — captions about the same event.  "
             "Visual similarity — similar-looking scenes.  "
             "Entity swaps — public figures in different rallies.", size=18)

    footer(s, "NewsCLIPpings: 39,288 balanced image–caption pairs built for a forensic challenge.")


def slide_balanced(prs):
    s = base_slide(prs, "The data", "A 50:50 test by design")
    rows = [
        ("Train", "~25,000", "Model learning"),
        ("Validation", "7,024", "Threshold and temperature"),
        ("Test", "7,264", "3,632 matched · 3,632 mismatched"),
    ]
    for i, (name, count, note) in enumerate(rows):
        y = 2.4 + i * 1.05
        card(s, 0.9, y, 11.4, 0.9)
        t1 = textbox(s, 1.2, y + 0.2, 2.8, 0.5)
        para(t1, name, size=20, bold=True, first=True)
        t2 = textbox(s, 4.0, y + 0.2, 2.4, 0.5)
        para(t2, count, size=20, first=True)
        t3 = textbox(s, 6.6, y + 0.2, 5.3, 0.5)
        para(t3, note, size=17, color=GREY, first=True)

    card(s, 0.9, 5.7, 11.4, 0.95, fill=BLACK, line_color=BLACK)
    tf = textbox(s, 1.25, 5.9, 10.7, 0.7)
    para(tf, "Random chance is 50%. Guessing or memorising fails immediately.",
         size=21, bold=True, color=WHITE, first=True)


def slide_preprocessing(prs):
    s = base_slide(prs, "Preprocessing", "Sanitize, canonicalize, cache")
    groups = [
        ("IMAGE", [
            "EXIF removed — no camera serials or GPS",
            "RGBA and other formats → 3-channel RGB",
            "Bilinear resize to 224 × 224, CLIP mean/std",
        ]),
        ("TEXT", [
            "Unicode normalized — accents and byte variants",
            "BPE, capped at 77 tokens with SOS/EOS",
            "Rejects < 3 characters or files > 10 MB",
        ]),
        ("AUDIT", [
            "0% image or caption overlap across splits",
            "512-D vectors cached as .npy arrays",
            "Clean inputs. Faster, repeatable runs.",
        ]),
    ]
    for i, (label, items) in enumerate(groups):
        x = 0.9 + i * 3.9
        card(s, x, 2.35, 3.6, 3.8)
        tf = textbox(s, x + 0.25, 2.6, 3.1, 3.4)
        para(tf, label, size=16, bold=True, color=GREY, first=True, space_after=14)
        for item in items:
            para(tf, f"— {item}", size=16, space_after=14)

    tf = textbox(s, 0.9, 6.4, 11.4, 0.5)
    para(tf, "Caching cuts training from 15 hours to 2 minutes.", size=18, bold=True, first=True)


def slide_model_selection(prs):
    s = base_slide(prs, "Model selection", "Why candidate D-C won")
    rows = [
        ("A", "Cosine only", "1", "54.2%"),
        ("B", "Linear concatenation", "1,024", "61.8%"),
        ("C", "Linear interaction", "2,048", "64.5%"),
        ("D-B", "MLP on concatenation", "1,024", "66.1%"),
        ("D-C", "MLP on explicit interaction", "2,048", "69.20%"),
    ]
    heads = [0.9, 1.7, 5.7, 8.3]
    widths = [0.7, 3.8, 2.4, 4.0]
    labels = ["", "Candidate", "Dim", "Accuracy"]
    for c, label in enumerate(labels):
        shape = card(s, heads[c], 2.35, widths[c], 0.55, fill=BLACK, line_color=BLACK)
        tf = shape.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(tf, label, size=14, bold=True, color=WHITE, first=True)

    for r, row in enumerate(rows):
        y = 2.95 + r * 0.62
        for c, val in enumerate(row):
            fill = WHITE if row[0] == "D-C" else (CREAM if r % 2 else WHITE)
            shape = card(s, heads[c], y, widths[c], 0.58, fill=fill)
            tf = shape.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = 0
            para(tf, val, size=15, bold=(row[0] == "D-C"), first=True)

    tf = textbox(s, 0.9, 6.15, 11.4, 0.8)
    para(tf, "Explicit interaction [u ; v ; |u−v| ; u⊙v] gives the MLP direct distance signals.",
         size=19, bold=True, first=True)
    footer(s, "D-C: macro F1 69.18% · ROC-AUC 77.83% · selected on validation data.")


NEW_SLIDES = [
    slide_real_vs_fakes,
    slide_balanced,
    slide_preprocessing,
    slide_model_selection,
]


def _fix_stale_numbers(prs):
    """Refresh the conformance figure on the existing 'How we tested' slide."""
    changed = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for p in shape.text_frame.paragraphs:
                for run in p.runs:
                    if "411 conformance" in run.text:
                        run.text = run.text.replace("411 conformance", "414 conformance")
                        changed += 1
                    if "410 pass" in run.text:
                        run.text = run.text.replace("410 pass", "413 pass")
                        changed += 1
    return changed


def add_to(path: Path) -> tuple[int, str]:
    prs = Presentation(str(path))
    existing = "\n".join(
        sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame
    )
    if MARKER in existing:
        return 0, "data slides already present - no changes made"

    fixed = _fix_stale_numbers(prs)
    for fn in NEW_SLIDES:
        fn(prs)

    ids = list(prs.slides._sldIdLst)
    new = ids[-len(NEW_SLIDES):]
    for element in new:
        prs.slides._sldIdLst.remove(element)

    # Three data slides after slide 4 (0-based index 4).
    for offset, element in enumerate(new[:3]):
        prs.slides._sldIdLst.insert(4 + offset, element)
    # Model selection after the D-C head slide, which is now index 12.
    prs.slides._sldIdLst.insert(13, new[3])

    prs.save(str(path))
    return len(new), f"inserted {len(new)} slides; refreshed {fixed} stale figures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, default=ROOT / "Project_Presentation.pptx")
    args = parser.parse_args()
    count, message = add_to(args.deck)
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
