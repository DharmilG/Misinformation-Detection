"""Generate the project presentation as a native PowerPoint deck.

Cream / black / white. One idea per slide. Plain words.
The architecture diagram is built from native shapes so it stays editable.

Usage:
    python scripts/build_presentation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent

# ── Palette ────────────────────────────────────────────────────────────────
CREAM = RGBColor(0xFB, 0xF8, 0xF1)
BLACK = RGBColor(0x11, 0x11, 0x11)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x55, 0x50, 0x4A)
LIGHT = RGBColor(0xE9, 0xE3, 0xD8)

FONT = "Aptos"

PROJECT_NAME = "Multimodal Out-of-Context Misinformation Detection Using Retrieval-Augmented Generation (RAG)"
MEMBERS = [
    "Dhaval Lakum",
    "Dharmil Gajjar",
    "Dhruv Chauhan",
    "Dhruv Mandaliya",
    "Drashti Patel",
    "Stuti Patel",
]


# ── Low-level helpers ──────────────────────────────────────────────────────

def set_bg(slide, color=CREAM):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_fade(slide):
    """Inject a simple fade transition into the slide XML."""
    sld = slide._element
    for old in sld.findall(qn("p:transition")):
        sld.remove(old)
    transition = sld.makeelement(qn("p:transition"), {"spd": "slow"})
    transition.append(sld.makeelement(qn("p:fade"), {}))
    # p:transition must follow p:cSld / p:clrMapOvr in schema order.
    insert_at = len(sld)
    for i, child in enumerate(sld):
        if child.tag in (qn("p:timing"), qn("p:extLst")):
            insert_at = i
            break
    sld.insert(insert_at, transition)


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, *, size=24, bold=False, color=BLACK, space_after=10,
         align=PP_ALIGN.LEFT, first=False, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = 1.15
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return p


def rule(slide, x, y, w, *, color=BLACK, height=0.02):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def card(slide, x, y, w, h, *, fill=WHITE, line_color=LIGHT):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line_color
    shape.line.width = Pt(0.75)
    return shape


def base_slide(prs, eyebrow, title, *, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    tf = textbox(slide, 0.85, 0.55, 11.6, 1.35)
    para(tf, eyebrow.upper(), size=13, bold=True, color=GREY, space_after=8, first=True)
    para(tf, title, size=40, bold=True, space_after=4)
    if subtitle:
        para(tf, subtitle, size=16, color=GREY, space_after=0)
        rule(slide, 0.85, 2.15, 11.6)
    else:
        rule(slide, 0.85, 1.95, 11.6)
    add_fade(slide)
    return slide


def footer(slide, text):
    tf = textbox(slide, 0.85, 7.05, 11.6, 0.3)
    para(tf, text, size=11, color=GREY, first=True)


def numbered(slide, items, *, x=1.0, y=2.35, w=11.2, size=22, gap=0.14):
    tf = textbox(slide, x, y, w, 4.4)
    for i, item in enumerate(items, 1):
        para(tf, f"{i}.  {item}", size=size, space_after=16, first=(i == 1))


# ── Slides ─────────────────────────────────────────────────────────────────

def slide_intro(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)

    rule(slide, 0.9, 0.75, 1.1, color=BLACK, height=0.06)

    tf = textbox(slide, 0.9, 1.05, 11.4, 0.45)
    para(tf, "SEMESTER 07  ·  BATCH ICT", size=14, bold=True, color=GREY, first=True)

    tf_title = textbox(slide, 0.9, 1.6, 11.4, 2.0)
    para(tf_title, "Multimodal Out-of-Context Misinformation", size=32, bold=True,
         first=True, space_after=2)
    para(tf_title, "Detection Using Retrieval-Augmented", size=32, bold=True, space_after=2)
    para(tf_title, "Generation (RAG)", size=32, bold=True, space_after=0)

    tf2 = textbox(slide, 0.9, 3.75, 11.4, 0.6)
    para(tf2, "A photo and a caption. Do they belong together?", size=23, color=GREY, first=True)

    rule(slide, 0.9, 4.5, 11.4)

    tf3 = textbox(slide, 0.9, 4.75, 11.4, 2.0)
    para(tf3, "TEAM", size=13, bold=True, color=GREY, space_after=12, first=True)
    # two columns of three names
    left = "\n".join(MEMBERS[:3])
    right = "\n".join(MEMBERS[3:])
    tb = slide.shapes.add_textbox(Inches(0.9), Inches(5.15), Inches(5.5), Inches(1.7))
    t = tb.text_frame
    t.word_wrap = True
    for i, line in enumerate(left.split("\n")):
        para(t, line, size=19, space_after=6, first=(i == 0))
    tb2 = slide.shapes.add_textbox(Inches(6.4), Inches(5.15), Inches(5.5), Inches(1.7))
    t2 = tb2.text_frame
    t2.word_wrap = True
    for i, line in enumerate(right.split("\n")):
        para(t2, line, size=19, space_after=6, first=(i == 0))

    add_fade(slide)


def slide_problem(prs):
    s = base_slide(prs, "The problem", "Real photo. Wrong caption.")
    card(s, 0.9, 2.4, 11.4, 2.5)
    tf = textbox(s, 1.3, 2.75, 10.6, 2.0)
    para(tf, "A photograph of a real event is reused with a caption about a different one.",
         size=26, bold=True, first=True, space_after=12)
    para(tf, "It looks believable. It spreads quickly. Checking it by hand is slow.",
         size=22, color=GREY)
    tf2 = textbox(s, 0.9, 5.4, 11.4, 1.0)
    para(tf2, "We build a tool that flags these pairs so a person can decide.",
         size=24, bold=True, first=True)
    footer(s, "Example: an image of one public event paired with a caption from another.")


def slide_why(prs):
    s = base_slide(prs, "Why it matters", "Help the checker. Do not replace them.")
    numbered(s, [
        "The score is a signal, not a verdict — about 3 in 10 answers are wrong.",
        "A fact-checker sees the flag, the evidence, and the reason.",
        "The human makes the final call and records it.",
    ])
    card(s, 0.9, 5.4, 11.4, 1.1, fill=BLACK, line_color=BLACK)
    tf = textbox(s, 1.25, 5.65, 10.7, 0.7)
    para(tf, "At this accuracy, automatic takedowns would be unsafe.",
         size=22, bold=True, color=WHITE, first=True)


def slide_built(prs):
    s = base_slide(prs, "What we built", "One tool, four jobs.")
    jobs = [
        ("CHECK", "Does the caption fit the image?"),
        ("EXPLAIN", "Show the score and the reason."),
        ("FIND", "Look for real supporting evidence."),
        ("REPORT", "Write it down so it can be audited."),
    ]
    for i, (label, desc) in enumerate(jobs):
        y = 2.35 + i * 1.1
        card(s, 0.9, y, 11.4, 0.95)
        tf = textbox(s, 1.2, y + 0.22, 2.3, 0.5)
        para(tf, label, size=18, bold=True, color=GREY, first=True)
        tf2 = textbox(s, 3.5, y + 0.18, 8.4, 0.6)
        para(tf2, desc, size=22, first=True)


def slide_architecture(prs):
    s = base_slide(prs, "Architecture", "How a request flows",
                   subtitle="Diagram below. The next slide explains each box.")

    def box(x, y, w, h, label, *, fill=WHITE):
        shape = card(s, x, y, w, h, fill=fill)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        para(tf, label, size=13, bold=True, align=PP_ALIGN.CENTER, first=True)

    def arrow(x1, y1, x2, y2):
        line = s.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
        )
        line.line.color.rgb = BLACK
        line.line.width = Pt(1.5)
        line.line.end_arrowhead = True

    # top row
    top = [
        (0.9, "1  Input\nimage + caption"),
        (3.0, "2  Validate\nfile checks"),
        (5.1, "3  CLIP\nembeddings"),
        (7.2, "4  D-C head\nMATCH / MISMATCH"),
        (9.3, "5  Calibrate\nconfidence"),
    ]
    for x, label in top:
        box(x, 2.5, 2.0, 1.15, label)
    for i in range(len(top) - 1):
        arrow(top[i][0] + 2.0, 3.075, top[i + 1][0], 3.075)

    # drop to RAG row
    arrow(10.3, 3.65, 10.3, 4.35)

    bottom = [
        (9.3, "6  Retrieve\nevidence"),
        (7.2, "7  Assess\nclaim vs passages"),
        (5.1, "8  Outcome\nSUPPORTED / …"),
    ]
    for x, label in bottom:
        box(x, 4.35, 2.0, 1.15, label)
    for i in range(len(bottom) - 1):
        arrow(bottom[i][0], 4.925, bottom[i + 1][0] + 2.0, 4.925)

    arrow(5.1, 4.925, 3.4, 4.925)
    box(0.9, 4.35, 2.5, 1.15, "9  Report\ntrace + citations", fill=BLACK)
    # recolor text of black box to white
    for shape in s.shapes:
        if shape.has_text_frame and "9  Report" in shape.text_frame.text:
            for p in shape.text_frame.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = WHITE

    tf = textbox(s, 0.9, 6.1, 11.4, 0.6)
    para(tf, "The RAG branch is optional. The classifier result never uses retrieved text.",
         size=16, color=GREY, italic=True, first=True)


def slide_arch_explain(prs):
    s = base_slide(prs, "Architecture explained", "What each box is for")
    numbered(s, [
        "Input takes the image and caption exactly as written.",
        "Validation rejects bad files before any model runs.",
        "CLIP turns both into vectors in one shared space.",
        "The D-C head compares them and makes the first decision.",
        "Calibration turns the score into a probability you can read.",
        "Retrieval finds real captions that may support or contradict the claim.",
        "Assessment compares the claim with those passages.",
        "Outcome names what the evidence says; inconclusive is allowed.",
        "The report stores the score, evidence ids and trace so the call can be reviewed.",
    ], size=18, gap=0.05)


def slide_classifier(prs):
    s = base_slide(prs, "The classifier", "Three numbers worth remembering")
    stats = [
        ("69.20%", "Test accuracy", "5,027 of 7,264 unseen pairs"),
        ("69.18%", "Macro F1", "balanced across both classes"),
        ("77.83%", "ROC AUC", "ranking quality, threshold-free"),
    ]
    for i, (big, label, note) in enumerate(stats):
        x = 0.9 + i * 3.9
        card(s, x, 2.5, 3.6, 2.7)
        tf = textbox(s, x + 0.25, 2.8, 3.1, 2.2)
        para(tf, big, size=40, bold=True, first=True, space_after=6)
        para(tf, label, size=20, bold=True, space_after=8)
        para(tf, note, size=15, color=GREY)
    tf2 = textbox(s, 0.9, 5.7, 11.4, 0.8)
    para(tf2, "Decision threshold 0.4507. Validation macro F1 68.71%. Best epoch 10, seed 42.",
         size=18, color=GREY, first=True)


def slide_errors(prs):
    s = base_slide(prs, "The errors", "It catches most fake pairs — not all")
    rows = [
        ("", "Predicted MATCHED", "Predicted MISMATCHED", "Total"),
        ("Actually MATCHED", "2,404  TN", "1,228  FP", "3,632"),
        ("Actually MISMATCHED", "1,009  FN", "2,623  TP", "3,632"),
        ("Total", "3,413", "3,851", "7,264"),
    ]
    lefts = [0.9, 3.6, 6.9, 10.3]
    widths = [2.6, 3.2, 3.3, 2.0]
    for r, row in enumerate(rows):
        y = 2.5 + r * 0.75
        for c, val in enumerate(row):
            fill = BLACK if r == 0 else (WHITE if r % 2 else CREAM)
            shape = card(s, lefts[c], y, widths[c], 0.7, fill=fill,
                         line_color=BLACK if r == 0 else LIGHT)
            tf = shape.text_frame
            tf.margin_left = tf.margin_right = Inches(0.08)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            para(tf, val, size=15, bold=(r == 0 or c == 0),
                 color=WHITE if r == 0 else BLACK, first=True)
    tf = textbox(s, 0.9, 5.8, 11.4, 1.0)
    para(tf, "Recall on misinformation 72.22%  ·  Precision 68.11%  ·  Specificity 66.19%",
         size=20, bold=True, first=True)
    para(tf, "Read plain: it catches about 7 in 10 fake pairs, and 1 in 4 flagged pairs are false alarms.",
         size=16, color=GREY)


def slide_testing(prs):
    s = base_slide(prs, "How we tested", "We tested the rules, not just the code")
    numbered(s, [
        "411 conformance cases. 410 pass. One expected skip. Zero failures.",
        "The encoder revision, checkpoint hash and temperature are recorded.",
        "The held-out test set was scored once and never used for tuning.",
        "The RAG protocol was written before the experiment, including the decision rule.",
    ])
    card(s, 0.9, 5.3, 11.4, 1.1)
    tf = textbox(s, 1.25, 5.55, 10.7, 0.7)
    para(tf, "A clean protocol lets us report a negative result without embarrassment.",
         size=20, bold=True, first=True)


def slide_rag_question(prs):
    s = base_slide(prs, "The RAG question", "Does evidence actually help?")
    card(s, 0.9, 2.4, 11.4, 2.2)
    tf = textbox(s, 1.3, 2.7, 10.6, 1.8)
    para(tf, "We fixed the rules first:", size=22, bold=True, first=True, space_after=10)
    para(tf, "primary metric = paired accuracy change · 1,000 bootstrap draws · "
             "inconclusive if the interval crosses zero", size=18, color=GREY)
    tf2 = textbox(s, 0.9, 5.1, 11.4, 1.2)
    para(tf2, "Then we ran one experiment. No tuning afterwards.", size=24, bold=True, first=True)


def slide_search_works(prs):
    s = base_slide(prs, "First surprise", "The search was fine")
    stats = [("71.43%", "Recall@5 — the right caption was found in 5 tries"),
             ("14.29%", "Precision@5 — most returned passages were noise"),
             ("0%", "Empty or invented passages")]
    for i, (big, label) in enumerate(stats):
        x = 0.9 + i * 3.9
        card(s, x, 2.5, 3.6, 2.2)
        tf = textbox(s, x + 0.25, 2.8, 3.1, 1.7)
        para(tf, big, size=36, bold=True, first=True, space_after=8)
        para(tf, label, size=16, color=GREY)
    tf2 = textbox(s, 0.9, 5.3, 11.4, 1.2)
    para(tf2, "For several wrong pairs, the true caption was rank 1. Retrieval was not the bottleneck.",
         size=22, bold=True, first=True)


def slide_judge_fails(prs):
    s = base_slide(prs, "Second surprise", "The judge never saw the photo")
    card(s, 0.9, 2.4, 11.4, 2.4)
    tf = textbox(s, 1.3, 2.7, 10.6, 2.0)
    para(tf, "The local judge answered SUPPORTED on all 10 claims.",
         size=26, bold=True, first=True, space_after=12)
    para(tf, "Five of them were genuinely false. It read the words. It could not see the image.",
         size=21, color=GREY)
    tf2 = textbox(s, 0.9, 5.3, 11.4, 1.1)
    para(tf2, "A judge that always agrees is not judging. We call this a degenerate assessor.",
         size=22, bold=True, first=True)


def slide_same_score(prs):
    s = base_slide(prs, "The core result", "Same score. Different answers.")
    stats = [("50%", "non-RAG accuracy"), ("50%", "RAG accuracy"),
             ("0%", "agreement between them")]
    for i, (big, label) in enumerate(stats):
        x = 0.9 + i * 3.9
        card(s, x, 2.4, 3.6, 1.9)
        tf = textbox(s, x + 0.25, 2.65, 3.1, 1.5)
        para(tf, big, size=38, bold=True, first=True, space_after=6)
        para(tf, label, size=16, color=GREY)
    card(s, 0.9, 4.6, 11.4, 1.9)
    tf = textbox(s, 1.3, 4.85, 10.6, 1.6)
    para(tf, "Change 0.0  ·  95% interval [−0.60, +0.60]  ·  verdict INCONCLUSIVE",
         size=20, bold=True, first=True, space_after=10)
    para(tf, "Ten questions, 5 true and 5 false. One student answered 'false' to all. "
             "The other answered 'true' to all. Both scored 5/10.",
         size=17, color=GREY)


def slide_why_flipped(prs):
    s = base_slide(prs, "Why it flipped", "Five small steps, one wrong habit")
    numbered(s, [
        "The evidence library is a bag of real captions, each tied to some photo.",
        "A false caption still lives in that bag — attached to a different photo.",
        "Search finds the caption itself at rank 1, because the words match.",
        "The judge sees only text, so 'the words exist' looks like 'supported'.",
        "SUPPORTED becomes MATCHED. All ten answers change.",
    ], size=19)
    card(s, 0.9, 5.7, 11.4, 0.95, fill=BLACK, line_color=BLACK)
    tf = textbox(s, 1.2, 5.9, 10.9, 0.7)
    para(tf, "Example: it cited 77605; the true caption for that image was 107208.",
         size=18, bold=True, color=WHITE, first=True)


def slide_learned(prs):
    s = base_slide(prs, "What we learned", "The useful result is where it breaks")
    numbered(s, [
        "We found three hidden bugs: fake search, mismatched data formats, an impossible threshold.",
        "Each was fixed and pinned with a test, so it cannot quietly return.",
        "Retrieval works; claim assessment is the part that needs real work.",
        "A clean inconclusive result is worth more than a win we did not earn.",
    ])
    card(s, 0.9, 5.4, 11.4, 1.0)
    tf = textbox(s, 1.25, 5.62, 10.7, 0.7)
    para(tf, "Search works. Judging doesn't. We know where to fix.", size=22, bold=True, first=True)


def slide_next(prs):
    s = base_slide(prs, "What's next", "Honest limits, clear next step")
    tf = textbox(s, 0.9, 2.3, 5.6, 0.5)
    para(tf, "LIMITS", size=14, bold=True, color=GREY, first=True)
    tf2 = textbox(s, 0.9, 2.75, 5.6, 3.6)
    for i, line in enumerate([
        "10 cases, one reviewer",
        "Closed-world corpus of dataset captions",
        "3 cases have no gold evidence",
        "No human-subject study yet",
        "No release archive or deployment monitoring",
    ]):
        para(tf2, f"— {line}", size=17, space_after=12, first=(i == 0))

    tf3 = textbox(s, 6.9, 2.3, 5.6, 0.5)
    para(tf3, "NEXT", size=14, bold=True, color=GREY, first=True)
    tf4 = textbox(s, 6.9, 2.75, 5.6, 3.6)
    for i, line in enumerate([
        "A new pre-registered study with a judge that can actually disagree",
        "100–300 cases with a second annotator",
        "Package and verify the checkpoint",
        "Then decide if RAG earns a place in the product",
    ]):
        para(tf4, f"— {line}", size=17, space_after=14, first=(i == 0))

    rule(s, 0.9, 6.5, 11.4)
    tf5 = textbox(s, 0.9, 6.65, 11.4, 0.5)
    para(tf5, "Thank you.", size=24, bold=True, first=True)


# ── Build ──────────────────────────────────────────────────────────────────

BUILDERS = [
    slide_intro, slide_problem, slide_why, slide_built,
    slide_architecture, slide_arch_explain, slide_classifier, slide_errors,
    slide_testing, slide_rag_question, slide_search_works, slide_judge_fails,
    slide_same_score, slide_why_flipped, slide_learned, slide_next,
]


def build(out: Path) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    for fn in BUILDERS:
        fn(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "Project_Presentation.pptx"
    )
    args = parser.parse_args()
    path = build(args.out)
    print(f"Wrote {len(BUILDERS)} slides to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
