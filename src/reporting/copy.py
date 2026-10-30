"""
src/reporting/copy.py
=====================
Every user-facing string in one reviewed module.

Spec: 05_REPORTING_AND_UI.md §9.2

Rules:
  - No user-visible wording is written inline in rendering code.
  - Changing any phrase is a single, reviewable edit here.
  - Forbidden verdict phrasings are listed at the bottom for lint tests.

Warning messages follow §13 exactly (text verbatim from the spec table).
"""
from __future__ import annotations

# ── Section headings ───────────────────────────────────────────────────────
H_ANALYZE           = "Analyze"
H_REPORT            = "Analysis Report"
H_EVIDENCE          = "Evidence Sources"
H_METHODOLOGY       = "Methodology & Limitations"
H_PRIVACY           = "Privacy & Data Handling"

H_CLASSIFIER        = "Image–Caption Consistency"
H_ASSESSMENT        = "Claim Assessment"
H_COMPARISON        = "Signal Comparison"
H_INPUT             = "Input"
H_RETRIEVED         = "Retrieved Evidence"
H_ACCEPTED_CITS     = "Accepted Citations"
H_REJECTED_CITS     = "Rejected Citations"
H_SYSTEM_LIMS       = "System Limitations"
H_PROVENANCE        = "Analysis Provenance"

# ── Classifier copy (§5) ──────────────────────────────────────────────────
CLASSIFIER_WHAT_MEASURES = (
    "This score measures how similar the image and caption are in the CLIP embedding space. "
    "It reflects visual–textual consistency, not factual claim accuracy."
)
# §9.1 level 1 — always visible, never collapsed
NOT_A_TRUTH_CLAIM = (
    "A mismatch score is not proof that the claim is false, "
    "and a match score is not proof that it is true."
)
SCORE_NOTE   = "an uncalibrated model output"
CAL_PROB_NOTE = "Calibrated probability"
CAL_NOT_APPLIED = "Calibration not applied to this result."

# ── Assessment copy (§7) ──────────────────────────────────────────────────
# §9.1 level 2 — always visible, never collapsed
SCOPE_QUALIFIER = (
    "This outcome reflects a system-computed result over model-classified evidence "
    "retrieved from a small local corpus. "
    "It does not establish the truth or falsity of the claim."
)
# §7.2 — shown whenever at least one citation is present
CITATION_SEMANTICS_DISCLAIMER = (
    "A citation marked here was located in the retrieved source and quoted verbatim. "
    "Whether it genuinely supports the claim is a judgement made by the model, "
    "and the outcome is a system-computed result from that judgement — "
    "not a verified determination."
)

# ── Evidence copy (§6) ────────────────────────────────────────────────────
# §6.3 — exact wording required when nothing was retrieved
EMPTY_EVIDENCE = (
    "The local corpus did not return evidence that met the retrieval criteria for this claim. "
    "This does not establish that relevant evidence does not exist elsewhere."
)
NO_EVIDENCE_SUGGESTION = (
    "Captions that include a name, a date, or a place are more likely to match "
    "a source in the local corpus."
)
NO_DATE    = "Publication date not stated by the source"
NO_URL     = "Source URL not available."
RELEVANCE_LABEL = "a ranking value, not a probability"

# Ordinal position labels (§6.2)
def rank_label(rank: int, total: int) -> str:
    """Return e.g. 'Top match (1 of 3 retrieved)'."""
    if rank == 1:
        position = "Top match"
        relative  = "The most relevant of the retrieved passages"
    elif rank == 2:
        position = "2nd retrieved"
        relative  = "The 2nd most relevant of the retrieved passages"
    elif rank == 3:
        position = "3rd retrieved"
        relative  = "The 3rd most relevant of the retrieved passages"
    else:
        position = f"{rank}th retrieved"
        relative  = f"The {rank}th most relevant of the retrieved passages"
    return f"{position} of {total} — {relative}"


# Storage scope (§6.1) — appears ONLY when storage_scope records it
STORAGE_EXCERPT   = "Only an excerpt is stored for this source."
STORAGE_CITATION  = "Only the citation metadata is stored for this source."

# ── Outcome icons (§16) ───────────────────────────────────────────────────
# Word + icon, never colour alone
OUTCOME_ICON: dict[str, str] = {
    "SUPPORTED":    "✓ Supported",
    "CONTRADICTED": "✗ Contradicted",
    "INCONCLUSIVE": "? Inconclusive",
    "CONFLICTING":  "⚠ Conflicting",
    "MISMATCHED":   "⚠ Mismatched",
    "MATCHED":      "✓ Matched",
}

# ── Warning messages (§13) ────────────────────────────────────────────────
WARNING_MESSAGES: dict[str, str] = {
    "IMAGE_DOWNSCALED":           "The image was resized before comparison.",
    "EXIF_ROTATED":               "The image was rotated to match its recorded orientation.",
    "ALPHA_FLATTENED":            "Transparency was replaced with a white background.",
    "TEXT_NORMALIZED":            "The caption's spacing and character forms were normalized.",
    "CAPTION_TRUNCATED_FOR_CLIP": (
        "The caption is longer than the comparison model's token limit, "
        "so only its beginning was used for the image comparison. "
        "The full caption was used for evidence retrieval."
    ),
    "LOW_CROP_RETENTION": (
        "The image's centre crop discards part of the image, "
        "so some visual context is not part of the comparison."
    ),
    "CLASSIFIER_UNAVAILABLE": "The image–caption comparison was not completed.",
}

# ── Input section ─────────────────────────────────────────────────────────
INPUT_ACCEPTED  = "Input accepted."
INPUT_REJECTED  = "Input rejected."

# ── Session copy (§12.5) ──────────────────────────────────────────────────
SESSION_CLEAR_DISCLOSURE = (
    "Clearing the session removes the report from this session. "
    "The application does not write reports to disk."
)

# ── Export disclosure (§15.4) ─────────────────────────────────────────────
EXPORT_DISCLOSURE = (
    "This file contains: the submitted caption, the image filename and dimensions, "
    "source titles and URLs, classifier and assessment outcomes, "
    "all retrieved passages, accepted and rejected citations, "
    "the comparison state, system limitations, and analysis provenance. "
    "It does not include the uploaded image file itself, "
    "and it does not include dataset-split or leakage diagnostics."
)

# ── Privacy copy (§17) ────────────────────────────────────────────────────
PRIVACY_LOCAL_INFERENCE = (
    "The uploaded image and caption are processed locally by the analysis pipeline. "
    "They are not sent to a remote inference provider."
)
PRIVACY_MODEL_DOWNLOADS = (
    "On first use, model weights may be downloaded from the internet "
    "unless they are already present in the local model cache. "
    "Subsequent runs reuse the cached weights."
)
PRIVACY_SOURCE_LINKS = (
    "Opening a source URL in the Evidence Sources section contacts that external website. "
    "This application does not control the privacy practices of external sites."
)
PRIVACY_STORAGE = (
    "Uploaded image bytes and the analysis report are held in the application session, "
    "in memory. The application does not write reports to disk. "
    "Model library caches written to disk are managed by the model library, "
    "not by this application."
)
PRIVACY_NO_TELEMETRY = (
    "This application does not send analytics, crash reports, usage counters, "
    "or any other outbound requests beyond those described above."
)
PRIVACY_HOSTED_BACKEND = (
    "No hosted assessment backend is currently configured. "
    "If one were configured, the following fields would be transmitted: "
    "the caption, retrieved passage text, source name, type, and date, "
    "and the classifier result. "
    "The image file, source URLs, and local caches would not be transmitted."
)

# ── Methodology copy ──────────────────────────────────────────────────────
METHODOLOGY_CLIP = (
    "The image–caption consistency signal uses CLIP "
    "(Contrastive Language–Image Pretraining). "
    "CLIP computes a cosine similarity between the image and the caption "
    "in a shared embedding space. "
    "A score below 0.5 is labelled MISMATCHED; at or above 0.5 is labelled MATCHED. "
    "This measures semantic similarity, not factual accuracy."
)
METHODOLOGY_RAG = (
    "The claim assessment uses a retrieval-augmented approach. "
    "The caption is used to retrieve relevant passages from a small local corpus. "
    "Retrieved passages are assessed for whether they support or contradict the claim. "
    "The assessment is a system-computed result over model-classified evidence, "
    "not a human judgment."
)
METHODOLOGY_LIMITATIONS: tuple[str, ...] = (
    "The local corpus is small and curated. "
    "Claims about topics not covered by the corpus will yield no evidence.",

    "The CLIP model measures semantic similarity, not factual accuracy. "
    "A matched pair may still contain a false claim.",

    "The claim assessment may misclassify citation relevance.",

    "Neither signal produces a verdict on whether a claim is true or false.",

    "Two signals pointing the same direction does not make either one correct.",

    "This tool is intended for research demonstration, not operational fact-checking.",
)

# ── Sample selector ───────────────────────────────────────────────────────
SAMPLE_SYNTHETIC_LABEL = "⚗ Synthetic fixture — test data, not evaluation data"
SAMPLE_UPLOAD_PRIORITY = (
    "You have uploaded an image. "
    "The uploaded image takes priority over the selected sample."
)

# ── Forbidden phrasings for lint tests (§9.3) ────────────────────────────
# Any string in this list appearing in rendered report text is a test failure.
FORBIDDEN_VERDICT_PHRASES: tuple[str, ...] = (
    "verdict: the claim is true",
    "verdict: the claim is false",
    "verdict: real",
    "verdict: fake",
    "overall verdict",
    "confirmed fake",
    "confirmed real",
    "claim is confirmed",
    "claim is verified",
)

FORBIDDEN_COMPARISON_WORDS: tuple[str, ...] = (
    "agree",
    "confirm",
    "verified",
    "prove",
)
