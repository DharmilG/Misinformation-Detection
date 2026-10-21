# 05 - Reporting and User Interface

## Multimodal Out-of-Context Misinformation Detection Using Retrieval-Augmented Generation (RAG)

**Document status:** Accepted - implementation contract for Phase 8  
**Document type:** Component specification  
**Depends on:** `../00_PROJECT_OVERVIEW.md`, `04_INPUT_AND_INFERENCE_PIPELINE.md`, `06_RAG_NON_RAG_EVALUATION.md`  
**Consolidated from:** `09_REPORTING_AND_UI.md`  
**Last updated:** 2026-10-21

---

## 1. Purpose and boundary

Document 05 defines **what the user sees**. It assembles the outputs of documents 04 and 06 into a report and renders it.

| Concern | Owner |
|---|---|
| The validated request and its warnings | Document 04 |
| The classifier result, its three quantities, and the explanatory copy | Documents 03 and 04 |
| The retrieved passages and their metadata | Document 06 |
| The computed outcome, the citations, the uncertainties, the provenance | Document 06 |
| **The report structure, the rendering, the error and fallback views, the comparison between the two signals** | **This document** |
| The overall test procedure and the project report | Document 07 |

### 1.1 The boundary, stated precisely

Document 05 **displays and compares** the outputs of documents 04 through 06.

| It does | It does not |
|---|---|
| Display the classifier result as document 04 computed it | Repair, reinterpret, or re-score it |
| Display the evidence and the assessment as documents 06 and 06 produced them | Re-derive eligibility, re-classify a citation, or recompute an outcome |
| Compare the two results, descriptively | Create a new truth verdict, or override the evidence outcome |
| Show limitations, uncertainties, and failures | Hide a degraded component to make the page look tidier |

Document 05 computes exactly one thing that its upstream documents do not: the **descriptive relationship** between two already-computed results, defined in §8.

### 1.2 Explicit non-promises

| Non-promise | Why it matters |
|---|---|
| **A single overall verdict** | Document 00 §4.2 forbids collapsing the two results into one "fake/real" label |
| **A verdict that implies the truth of a claim** | The assessment is a system-computed outcome based on model-classified evidence (document 06 §2.1) |
| **A fabricated field** | Every displayed value comes from a stored record. Nothing is filled in, and no link is manufactured |
| **A claim that agreement implies correctness** | Two signals pointing the same way does not make either one correct |
| **A complete search** | The corpus is small and local. Absence of evidence is not evidence of absence |

---

## 2. The report model

### 2.1 Structure

```python
@dataclass(frozen=True)
class ReportModel:
    request_id: str
    analysis_fingerprint: str
    created_at: datetime
    input: InputSection
    classifier: ClassifierSection
    evidence: EvidenceSection
    assessment: AssessmentSection
    comparison: ComparisonSection
    limitations: tuple[str, ...]
    provenance: ReportProvenance
```

### 2.2 The per-section status contract

The interface introduces **no new status vocabulary**. It reuses `OK / UNAVAILABLE / SKIPPED / FAILED`, the vocabulary from `04_INPUT_AND_INFERENCE_PIPELINE.md` §11 and `06_RAG_NON_RAG_EVALUATION.md` §31, and each section's status is the upstream status it came from.

| Section | Status source |
|---|---|
| `InputSection` | Always `OK` — a report exists only for a valid request |
| `ClassifierSection` | Document 04, `ClassifierResult.status` |
| `EvidenceSection` | Document 06, `RetrievalResult.status` |
| `AssessmentSection` | Document 06, `Assessment.status` |
| `ComparisonSection` | Always constructible; its **state** varies — see §8 |

### 2.3 Required payload per section and status

| Section | `OK` payload | `UNAVAILABLE` payload | `SKIPPED` payload | `FAILED` payload |
|---|---|---|---|---|
| `InputSection` | The image reference, the caption, the warnings | — | — | — |
| `ClassifierSection` | Label, score, probability, calibration status, the copy | Reason, remedy where one exists | Reason | Reason, error detail |
| `EvidenceSection` | The passages, the counts, the corpus stats | Reason | Reason | Reason |
| `AssessmentSection` | Outcome, reason, scope text, components, citations, uncertainties, conflicts, limitations | Reason | Reason | Reason, error detail |
| `ComparisonSection` | The comparison state and its sentence | — | — | — |

### 2.4 Invariants

| Invariant | Asserted by test |
|---|---|
| A section's status equals the upstream status it came from. Document 05 never re-classifies it | Yes |
| `OK` ⟹ the required payload exists and passes schema validation | Yes |
| `UNAVAILABLE`, `SKIPPED`, `FAILED` ⟹ no payload is presented as a result | Yes |
| **`InputSection.status == OK` means the request passed input validation** — nothing about downstream success | Yes |
| **`ComparisonSection` always has a well-defined state.** "Always present" does not mean "always a meaningful comparison" | Yes |
| Every value rendered comes from a stored record | Yes |
| No rendered value is a placeholder — never `""`, `"N/A"`, `"unknown"`, or `"-"` | Yes |

### 2.5 Why the per-section status contract is the whole design

The renderer never asks "did everything succeed?" It renders each section according to that section's status. A page with a broken classifier, empty evidence, and a failed assessment is not a special case — it is four ordinary section states, and the renderer handles it with the same code that handles the fully successful case.

This is what makes document 00 §9.4's requirement — that a failed component must not silently generate a fabricated prediction or report — true by construction.

---

## 3. The two-signal summary

### 3.1 The summary strip

A completed analysis opens with a **two-panel strip**, side by side.

```text
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ IMAGE–CAPTION CONSISTENCY    │  │ CLAIM ASSESSMENT             │
│ Mismatch suspected           │  │ Inconclusive                 │
│ Score 0.82 (uncalibrated)    │  │ No eligible evidence         │
└──────────────────────────────┘  └──────────────────────────────┘
```

| Rule | Detail |
|---|---|
| Two panels, always both | There is no single headline verdict, and no combined badge |
| Each panel names its own scope | One is about the image–caption relationship. The other is about the claim |
| The panels never merge | Even when both succeed and point the same way, they remain two panels with two labels |
| A degraded panel stays visible | If the classifier is unavailable, its panel says so. It is not hidden |
| The strip is a summary, not a verdict | It points down to the sections, which carry the reasoning |

Glanceability without a false synthesis. A user sees both answers at once and cannot mistake either for "the verdict".

### 3.2 The panel contents

| Panel | Shows |
|---|---|
| Image–caption consistency | The label, the **score**, the calibrated probability when one exists, the calibration status, and the model's own explanation from `copy.json` |
| Claim assessment | The outcome, the reason code in plain words, and the scope qualifier |

The score and the probability are distinguished exactly as document 06 defines them: the score is the uncalibrated class probability, the probability is the calibrated one, and the calibration status is shown. A score with no calibration artifact is labelled as a score, not as a probability.

---

## 4. The input section

| Element | Rendering |
|---|---|
| The image | A preview, with descriptive alt text naming the file and its dimensions |
| The caption | The user's original text, exactly as submitted |
| The validation status | A short "Input accepted" line, since the page only exists for a valid request |
| The warnings | Each warning with its message, in the wording from §13 |

The caption is displayed verbatim, not normalized, so the user sees what they typed. The normalized form is internal.

---

## 5. The image–caption consistency section

| Element | Rendering |
|---|---|
| The label | The word, with the icon from §16. Never a colour alone |
| The score | "Score 0.82 — an uncalibrated model output" |
| The probability | "Calibrated probability 0.74", only when `calibration_status == APPLIED`. Otherwise the score carries a "not calibrated" note |
| The explanation | The model's own `what_this_measures` from `copy.json` |
| The limitation | The `not_a_truth_claim` string, **always visible**, never inside a collapsed expander |
| The provenance | The model version, the run id, the encoder revision, and the score's definition |

When the section is not `OK`, the panel shows the reason and the remedy, and **no score or label**.

---

## 6. The evidence section

### 6.1 The passage list

Each retrieved passage renders:

| Element | Rendering |
|---|---|
| The title | The stored title, escaped |
| The source | The source name and the source type |
| The date | The stored `published_at`, or **"Publication date not stated by the source"** |
| The excerpt | The retrieved passage text, escaped, truncated for display with the truncation noted |
| The URL | A clickable link when a valid source URL exists. Otherwise **"Source URL not available."** Never a manufactured link |
| The usage restriction | The stored `rights_wording`, when present. Otherwise it is not shown |
| The storage scope | "Only an excerpt is stored for this source" or "Only the citation metadata is stored for this source", **only when `storage_scope` records it** |
| The relevance | The rank and a relative position. A score, if shown, is labelled as a ranking value, not a probability |

### 6.2 Relevance, explained

| Element | Rendering |
|---|---|
| The position | "Top match", "2nd retrieved", and so on, with the ordinal number |
| The relative indicator | "The most relevant of the retrieved passages" — a plain-language position within the returned set |
| A score, if shown | "RRF relevance score 0.031 — a ranking value, not a probability" |

No bare number is ever shown in a way that reads as a confidence percentage.

### 6.3 When nothing was retrieved

> **"The local corpus did not return evidence that met the retrieval criteria for this claim. This does not establish that relevant evidence does not exist elsewhere."**

Followed by the corpus coverage — how many documents and passages the corpus holds — and the recommendation to check additional sources.

This wording is deliberate. The earlier phrasing, "No relevant evidence was found", risked implying that no relevant evidence exists at all, which the small local corpus cannot support.

### 6.4 Retrieved versus cited

The evidence section lists **all retrieved passages**. The assessment section lists **the citations that were accepted**, and, when there are any, **the citations that were rejected with their reasons**. The two lists are not merged, because a passage returned by retrieval is not necessarily a passage the assessment relied on.

---

## 7. The assessment section

| Element | Rendering |
|---|---|
| The outcome | The word and the icon. `SUPPORTED`, `CONTRADICTED`, `INCONCLUSIVE`, `CONFLICTING` |
| The reason | In plain words, from the four reason codes of document 06 §30.2 |
| The scope | The scope qualifier: "The retrieved evidence supports each examined component of the claim, within the scope of the retrieved corpus" — **always visible** |
| The claim components | Each component with its status — supported, contradicted, contested, or unresolved — and the citations attached to it |
| The citations | The quote, escaped, with the source, the date, and the link |
| The rejected citations | With the reason, so a fabrication attempt is visible rather than silently removed |
| The unresolved items | What the evidence did not settle |
| The conflicts | Where the sources disagree |
| The uncertainties | The model's explicit uncertainty |
| The limitations | What the evidence cannot establish |

### 7.1 The outcome wording never implies truth

The assessment panel states the **outcome of a computation over model-classified evidence**. The word "true" and "false" do not appear in relation to the claim.

### 7.2 The distinction between citation validity and semantic support

A citation appearing in the report means its identifier resolved and its quote was verbatim. It does **not** mean the passage was proven to support the claim. The report's wording reflects that:

> A citation marked here was located in the retrieved source and quoted verbatim. Whether it genuinely supports the claim is a judgement made by the model, and the outcome is a system-computed result from that judgement — not a verified determination.

This sentence appears in the assessment section whenever at least one citation is present.

---

## 8. The comparison section

### 8.1 The two directions

The classifier and the assessment measure **different things**. The comparison is therefore defined over directions, not over truth.

| Axis | Values |
|---|---|
| Classifier direction | `MISMATCH_INDICATED` (label `MISMATCHED`) · `NO_MISMATCH_INDICATED` (label `MATCHED`) · `UNAVAILABLE` (any non-`OK` status) |
| Assessment direction | `CONTRADICTS_CLAIM` (`CONTRADICTED`) · `SUPPORTS_CLAIM` (`SUPPORTED`) · `UNRESOLVED` (`CONFLICTING`, `INCONCLUSIVE`) · `UNAVAILABLE` (any non-`OK` status) |

### 8.2 The exhaustive mapping

| Classifier direction | Assessment direction | Comparison state |
|---|---|---|
| `MISMATCH_INDICATED` | `CONTRADICTS_CLAIM` | `CONSISTENT_PATTERN` |
| `MISMATCH_INDICATED` | `SUPPORTS_CLAIM` | `DIFFERENT_PATTERN` |
| `MISMATCH_INDICATED` | `UNRESOLVED` | `ASSESSMENT_UNRESOLVED` |
| `MISMATCH_INDICATED` | `UNAVAILABLE` | `ASSESSMENT_UNAVAILABLE` |
| `NO_MISMATCH_INDICATED` | `SUPPORTS_CLAIM` | `CONSISTENT_PATTERN` |
| `NO_MISMATCH_INDICATED` | `CONTRADICTS_CLAIM` | `DIFFERENT_PATTERN` |
| `NO_MISMATCH_INDICATED` | `UNRESOLVED` | `ASSESSMENT_UNRESOLVED` |
| `NO_MISMATCH_INDICATED` | `UNAVAILABLE` | `ASSESSMENT_UNAVAILABLE` |
| `UNAVAILABLE` | any | `CLASSIFIER_UNAVAILABLE` |

Five states, not four. `ASSESSMENT_UNAVAILABLE` is separated from `ASSESSMENT_UNRESOLVED` because *unavailable* and *unresolved* are different conditions throughout this project, and merging them here would undo documents 04 and 06.

`CLASSIFIER_UNAVAILABLE` is worded so that it does **not** imply the assessment is missing. The assessment panel stands unaffected.

### 8.3 The wording

| State | Sentence |
|---|---|
| `CONSISTENT_PATTERN` | "Both results point in the same direction: the comparison indicates a mismatch, and the retrieved evidence contradicts the claim." |
| `DIFFERENT_PATTERN` | "The two results point in different directions. The comparison indicates a mismatch, and the retrieved evidence supports the claim. These measure different things — one is about the image–caption relationship, the other is about the claim." |
| `ASSESSMENT_UNRESOLVED` | "The retrieved evidence did not settle the claim, so the two results cannot be compared." |
| `ASSESSMENT_UNAVAILABLE` | "No claim assessment was produced, so the two results cannot be compared." |
| `CLASSIFIER_UNAVAILABLE` | "The image–caption comparison was unavailable. The claim assessment below is unaffected." |

The words **"agree", "confirm", "verified", and "prove" do not appear** in any comparison sentence. The section describes a relationship between two outputs. It does not claim that either output was confirmed.

### 8.4 The permanent note

> **"Two signals pointing the same way does not make either one correct."**

It appears in the comparison section in every case where a comparison state is rendered.

### 8.5 A meaningful distinction, not a system conflict

A coherent image–caption pair whose claim is contradicted by reliable evidence yields `DIFFERENT_PATTERN`. That is a real and interesting situation — the image and text agree with each other, and the claim is still false — not a malfunction of either system. The wording in §8.3 says so explicitly.

---

## 9. Limitations and copy

### 9.1 Always visible, at three levels

| Level | Content | Source |
|---|---|---|
| Under the classifier result | "A mismatch score is not proof that the claim is false, and a match score is not proof that it is true." | `copy.json`'s `not_a_truth_claim` |
| Under the assessment | The scope qualifier — support for the examined components, within the retrieved corpus | Document 06 §30.5 |
| At the page level | The system's scope: what the tool does, what it does not do, that a small local corpus will often find nothing, and that two signals pointing the same way does not make either one correct | Assembled from the upstream documents' limitations |

**None of these is inside a collapsed expander.** A reader who expands nothing still sees what the results do and do not mean.

### 9.2 The copy is centralised

All user-facing strings live in one reviewed module. No user-visible wording is written inline in the rendering code, so a change to a phrase is a reviewable, versioned edit rather than a string buried in a widget.

### 9.3 The copy checks

A forbidden-phrase blacklist cannot prove that all wording is safe — a different sentence can still imply a definitive verdict. Four layers are used together:

| Layer | What it does |
|---|---|
| Centralised, reviewed copy | One module, reviewed by a person |
| Forbidden-wording tests | A lint rejects explicitly forbidden phrasings such as "the claim is true", "the claim is false", "confirmed", and "verified" in a verdict context |
| Scope-distinctness tests | The classifier panel never contains the assessment's scope text, and the assessment panel never contains the classifier's explanation. The two scopes stay visibly separate |
| Manual review | Representative successful and degraded reports are read by a person and recorded as a step in the acceptance criteria |

---

## 10. The error and fallback matrix

Every case from document 00 §9.4, with its rendering.

| Condition | Section | What the user sees |
|---|---|---|
| Missing or unsupported image | — | The **input-rejection view** (§11). No report exists |
| Corrupt upload | — | The input-rejection view |
| Empty or over-long caption | — | The input-rejection view |
| Missing classifier checkpoint | `ClassifierSection` | "The model is unavailable." + the reason + the remedy. **The evidence and assessment sections still render** |
| Model-loading failure | `ClassifierSection` | The same |
| Encoder unavailable or mismatched | `ClassifierSection` | The same, with the corresponding reason |
| **No relevant retrieved evidence** | `EvidenceSection`, `AssessmentSection` | The §6.3 wording + the corpus coverage. The assessment shows `INCONCLUSIVE` with `NO_EVIDENCE`. **No outcome is displayed** |
| Generation failure or rate limit | `AssessmentSection` | "No assessment could be generated." + the reason. **The retrieved sources still render** |
| Invalid or missing source metadata | `EvidenceSection` | The citation shows the fields that exist, and "Not stated" for the rest |

### 10.1 The rules behind the matrix

| Rule | Detail |
|---|---|
| A degraded section is **visible** | It is never hidden to make the page look tidier |
| A degraded section produces **no result values** | No label, no score, no outcome, no probability |
| A degraded section does **not** stop the others | The page renders in full, with the failures marked |
| The whole page renders | For a valid request, the page always renders |
| A failed component never produces a fabricated value | Document 00 §9.4 |

---

## 11. The input-rejection view

A **different view**, not a report. It exists when document 04 rejects the input.

| Element | Content |
|---|---|
| The message | The validation error's user-facing message, verbatim, from document 04 |
| The remedy | Where one exists — for example, the accepted formats, the size limit, or the caption length |
| The state of the interface | The uploader and the caption box remain available, and the report area shows nothing |

**No request exists**, so there is no `request_id`, no report model, and no model loading. The input-rejection view must not attempt to render a report with a failed input section, because that would imply the analysis ran.

---

## 12. Session state, identities, and reruns

### 12.1 Three distinct identities

| Concept | Identity | Lifetime |
|---|---|---|
| **Model registry identity** | Which loaded encoder and classifier instances are reused | Process lifetime — document 04's registry |
| **Analysis identity** | `request_id` — which run produced this report | One run |
| **Cache identity** | `analysis_fingerprint` — whether an earlier result is safe to reuse | Across reruns within a session |

### 12.2 The analysis fingerprint

```text
analysis_fingerprint = sha256(
    image_sha256
  + caption_sha256
  + model_version + run_id
  + encoder_revision + feature_contract_version
  + corpus_manifest_sha256
  + prompt_version
  + retrieval_config_hash
  + embedding_revision + chunking_version
)
```

A submission hash alone is not enough. If the model, the corpus, the prompt, or the retrieval configuration changes, the inputs may be identical while the correct output is not.

### 12.3 The rerun rules

| Action | Behaviour |
|---|---|
| Expand a citation, toggle a display option, open a source link | **No re-run.** The report is reused |
| Submit new input | A **new analysis** with a new `request_id` and a new fingerprint |
| An explicit rerun | A **fresh analysis** with a new `request_id`, per the documented policy. The fingerprint may be identical if nothing changed |
| The model, corpus, prompt, or retrieval configuration changes | The fingerprint changes, so the cached report is **not reused**. A stale report is never displayed |

The cached report is stored in the session state keyed by its fingerprint, alongside the fingerprint of the current submission.

### 12.4 One cache owner

Document 05 calls document 04's process-level registry and **adds no second cache** for the model artifacts. There is one owner of the loaded model and encoder instances, and therefore no possibility of two live copies.

### 12.5 Session cleanup

| Behaviour | Statement |
|---|---|
| A "Clear session" control | Removes the stored report and the selected sample input from the session |
| What it does not clear | The browser, the model caches, or anything the user has already copied or captured |
| Honest wording | "Clearing the session removes the report from this session. The application does not write reports to disk." |

Session state holds text and small records, not tensors, so a report is small enough to keep for the whole session.

---

## 13. The warnings

| Warning | Message | Constraint |
|---|---|---|
| `IMAGE_DOWNSCALED` | "The image was resized before comparison." | — |
| `EXIF_ROTATED` | "The image was rotated to match its recorded orientation." | — |
| `ALPHA_FLATTENED` | "Transparency was replaced with a white background." | — |
| `TEXT_NORMALIZED` | "The caption's spacing and character forms were normalized." | — |
| `CAPTION_TRUNCATED_FOR_CLIP` | "The caption is longer than the comparison model's token limit, so only its beginning was used for the image comparison. The full caption was used for evidence retrieval." | The second clause is accurate: document 06 uses the full normalized caption. The message must match the confirmed truncation behaviour |
| `LOW_CROP_RETENTION` | "The image's centre crop discards part of the image, so some visual context is not part of the comparison." | **Only** shown if the metric is actually computed and its meaning is documented. It never says a different image would compare better, and it never explains the prediction |
| `CLASSIFIER_UNAVAILABLE` | "The image–caption comparison was not completed." | — |

### 13.1 The constraint on actionable advice

Every message describes **the analysis**, not the cause of a prediction. None of them says that a different image or a different caption would produce a better result, because that is not something the system can know.

A suggestion is offered only where the remedy is genuinely within the user's control and genuinely likely to help — for example, "captions that include a name, a date, or a place are more likely to match a source in the local corpus" when the outcome is `NO_EVIDENCE`.

---

## 14. Sample inputs

### 14.1 The mechanism

A `st.file_uploader` value cannot be set programmatically, so the sample path is **separate from the uploader**.

| Property | Decision |
|---|---|
| Control | A "Use a sample input" selector, listing the synthetic fixture samples |
| Behaviour | Selecting one stores the fixture's image bytes and its caption in the session state, and populates the caption box |
| Bypasses | The uploader widget only |
| Does **not** bypass | Document 04's validation. The fixture bytes go through `build_input_request` exactly like a user's upload |
| Precedence | **A user upload takes priority.** If both a sample and an upload are present, the upload is used and the interface says so |
| The caption box | Populated normally and editable. An edit changes the caption, hence the fingerprint, hence a new analysis |

### 14.2 The source

Document 06's **synthetic** reference fixture only. It carries no third-party rights and no dataset content, so a demonstration works with no dataset and no network.

The interface labels the samples as synthetic. They are a test fixture, not a dataset, and they are not evaluation data.

---

## 15. Export

### 15.1 The format and the content

Markdown, copied to the clipboard and downloadable.

| Section | Content |
|---|---|
| The submission | The caption, the image filename and its dimensions |
| The image–caption consistency | The label, the score, the probability and the calibration status |
| The claim assessment | The outcome, the reason, the scope, the components, the citations, the uncertainties, the conflicts, the limitations |
| **The retrieved passages** | **All** of them, marked as retrieved |
| **The accepted citations** | Marked as cited, with the quote and the source |
| **The rejected citations** | With the reason, when there are any |
| The comparison | The state and the sentence |
| The limitations | All of them, from §9 |
| The provenance | The model version, the run id, the encoder revision, the prompt version, the corpus identity, the retrieval configuration, the timestamps, and the analysis fingerprint |

### 15.2 The retrieved-versus-cited distinction is preserved

A passage returned by document 06 is not necessarily a passage document 06 relied on. The export keeps the two lists separate, so a reader can see how many passages were retrieved, how many were cited, and why the others were not.

### 15.3 The dataset diagnostics are not in the per-analysis export

Leakage exposure and split diagnostics describe a **model's training data**, not an individual analysis. They belong to document 07's evaluation report, keyed to the model version, and they are **not** included here. A reader of one export gets the model-level caveats that apply to that analysis: the model version, the corpus identity, the scope text, and the assessment's own uncertainties.

### 15.4 Export privacy

| Rule | Detail |
|---|---|
| Explicit initiation | The user clicks export. Nothing is exported automatically |
| Disclosure before the download | The interface states what the file contains: the submitted caption, the image filename and dimensions, the source titles and URLs, and the provenance |
| The user initiates it | The user is the one exporting their own submission |
| Escaping | All untrusted text is escaped, so retrieved content cannot inject formatting into the exported document |

---

## 16. Accessibility and presentation

| Property | Decision |
|---|---|
| Outcome encoding | A **word and an icon**, never colour alone. Supported ✓, Contradicted ✗, Inconclusive ?, Conflicting ⚠ |
| Colour | Supplementary, and chosen for contrast in both light and dark themes |
| The image preview | Descriptive alt text naming the file and its dimensions |
| Section order | The same order as the report model, so the reading order is predictable |
| Text completeness | The full meaning of every result is available as text. Nothing is conveyed only by a graphic |
| Escaping | Every untrusted string is escaped, so retrieved content cannot alter the structure of the page |

### 16.1 Escaping untrusted text

Retrieved passages, quotes, titles, and source names are **untrusted** and are rendered into the page and into the export. Without escaping, Markdown in a passage could inject headings, links, or emphasis into the report.

| Rule | Detail |
|---|---|
| Every untrusted string is escaped | The caption, passage text, quotes, titles, and source names |
| Quotes | Rendered in an escaped block that does not interpret markup |
| The export | Escaped identically |
| A test | A fixture passage containing Markdown, unusual Unicode, and a control sequence renders as inert text and does not alter the report's structure |

---

## 17. Privacy

### 17.1 The contract

| Item | Accurate statement |
|---|---|
| **Local inference** | The uploaded image and caption are **not sent to a remote inference provider** by the analysis pipeline |
| **Initial setup** | Model downloads may require a network connection unless the required models are already cached |
| **Source links** | Opening a source URL contacts that external website |
| **Hosted inference** | The exact transmitted fields, enumerated in §17.2 |
| **Storage** | The uploaded bytes and the report are held in the application session, in memory. **The application writes no report to disk.** The model caches written by the model libraries are separate and outside the application's control |

The earlier blanket statement, "nothing leaves the machine", was too broad. It did not distinguish the analysis pipeline from setup downloads, from opening a source link, or from the model caches.

### 17.2 The hosted-backend disclosure

When the optional hosted backend from document 06 is configured, the interface states exactly what is transmitted. Derived from document 06 §26's prompt assembly:

| Transmitted | Not transmitted |
|---|---|
| The caption | The image file — it is processed locally to produce the classifier result |
| The retrieved passage text | The source URLs |
| The source name, type, and publication date | The retrieved passages that did not make the prompt |
| The classifier result and its interpretation | The model weights and the local caches |

The disclosure text is written to match the implementation. If the payload changes, the disclosure changes with it, and a test asserts that the disclosed field list matches the prompt assembly's field list.

### 17.3 No telemetry

No analytics, no crash reporting, no usage counters, no outbound requests other than those enumerated in §17.1.

---

## 18. The text-rendering layer

### 18.1 The design

Streamlit widgets are hard to unit-test, so the report is rendered to plain text by a **pure function** first.

```python
def render_report_text(report: ReportModel) -> str: ...
```

| Property | Decision |
|---|---|
| Pure | No Streamlit import, no I/O, no side effects |
| Complete | The text rendering contains everything the interface shows, so a test can assert on it |
| Streamlit is a thin wrapper | The widgets read the same `ReportModel` and display the same content. There is one source of truth for the wording |
| Testable | Every status combination, every copy string, and every null-metadata case is testable in milliseconds |

### 18.2 What this buys

The copy lint, the forbidden-wording checks, the scope-distinctness checks, the null-metadata checks, and the escaping checks all run against this function. Without it, those checks would require a running Streamlit server, and they would not run in a normal test suite.

---

## 19. Testing strategy

### 19.1 Test tiers and commands

The test tier taxonomy, the marker definitions, the environment requirements, the complete command set, the coverage gate, and the evaluation-harness invariants are specified centrally in `07_INTEGRATION_TESTING.md` §2 and §3.

The tests listed in the remaining subsections of this section are this component's own tests, executed under those tiers. They are **unit and component tests**: they verify this component's contracts, not the boundaries between components, which are tested in `07_INTEGRATION_TESTING.md` §5.

```bash
# This component's fast suite
pytest -m "not slow and not eval and not benchmark and not e2e"
```
### 19.2 Fast tests

| Test | Assertion |
|---|---|
| **Every status combination renders** | For each of the classifier, evidence, and assessment statuses, the text rendering is non-empty, contains the section heading, and contains no placeholder text |
| **Never a combined verdict** | The rendered text contains no single overall verdict. The copy lint rejects combined-verdict wording |
| **Limitations always present** | The page-level limitations and the classifier's `not_a_truth_claim` appear, for every status combination |
| **Scope distinctness** | The classifier panel does not contain the assessment's scope text, and the assessment panel does not contain the classifier's explanation |
| **Outcome in words** | The outcome appears as a word plus an icon, never as a colour alone |
| **Comparison matrix, exhaustive** | Every (classifier status and label) × (assessment status and outcome) combination yields a defined state and a sentence. No combination is undefined, and none is worded as agreement |
| **Comparison wording** | No comparison sentence contains "agree", "confirm", "verified", or "prove" |
| **Citation semantics** | The assessment's wording about citations does not claim semantic support. A valid identifier and a verbatim quote are not described as proving support |
| **Null dates** | A source with `published_at: null` renders "Publication date not stated by the source" |
| **Absent fields** | An absent metadata field renders "Not stated", never a blank and never a dash |
| **Missing and invalid URLs** | A passage with no valid URL renders "Source URL not available." **No link is manufactured** |
| **Storage scope** | "Only an excerpt is stored" appears only when `storage_scope` records it, and never on the basis of a usage restriction |
| **Empty evidence** | The §6.3 wording appears, the assessment shows `INCONCLUSIVE` with `NO_EVIDENCE`, and no outcome is rendered |
| **Unavailable classifier** | The section states the unavailability and the remedy, and the evidence and assessment sections still render |
| **Unavailable assessment** | The section states the failure, and the retrieved sources still render |
| **Sources are complete** | Every retrieved passage renders a title, a source name, an excerpt, and either a valid link or the explicit statement that none is available |
| **Relevance wording** | A score, when rendered, is labelled as a ranking value and not as a probability |
| **Comparison note present** | "Two signals pointing the same way does not make either one correct" appears in every comparison state |
| **Escaping** | A fixture passage containing Markdown, unusual Unicode, and a control sequence renders as inert text and does not alter the report's structure |
| **Copy lint** | The forbidden phrasings are rejected |
| **No colour-only encoding** | The rendered text carries the outcome as a word |
| **The registry is not duplicated** | The interface does not wrap the model artifacts in a second cache |
| **Cache invalidation** | Changing the model version, the corpus hash, the prompt version, or the retrieval configuration changes the fingerprint, so a cached report is not reused |
| **Rerun rules** | A display change leaves the `request_id` unchanged; a new submission produces a new `request_id`; an explicit rerun produces a new `request_id` |
| **Export privacy** | The exported Markdown contains exactly the enumerated fields, and no field outside the list |
| **Export distinctions** | The export separates retrieved passages from accepted citations and includes the rejected citations with their reasons |
| **Export excludes the dataset diagnostics** | The export contains no leakage or split diagnostic |
| **Session cleanup** | After the clear action, the stored report and the selected sample are gone, and the model registry is untouched |
| **Fixture loading** | Selecting a sample populates the caption, bypasses the uploader, and the bytes still pass document 04's validation |
| **Input rejection** | An invalid input renders the rejection view, and no report model is constructed |
| **The copy is centralised** | No user-visible string is defined inline in the rendering code |

### 19.3 End-to-end tests

| Test | Assertion |
|---|---|
| Full flow | A valid request renders all five sections with real content |
| Degraded classifier | The classifier section is marked unavailable, the rest renders, and no score appears |
| Degraded assessment | The assessment section is marked unavailable, the retrieved sources still render, and no outcome appears |
| Empty evidence | The evidence section shows the §6.3 wording and the assessment shows `INCONCLUSIVE` / `NO_EVIDENCE` |
| The comparison is rendered | The state and its sentence appear, with the permanent note |
| The export matches the page | The exported Markdown contains the same outcomes, citations, and limitations as the rendered page |

---

## 20. Acceptance criteria

Phase 8 is complete when:

1. The report structure is a `ReportModel` with a **per-section status**, and the renderer handles every status through the same code path.
2. The interface introduces **no new status vocabulary**; it reuses `OK / UNAVAILABLE / SKIPPED / FAILED` from `04_INPUT_AND_INFERENCE_PIPELINE.md` §11 and `06_RAG_NON_RAG_EVALUATION.md` §31, and each section's status is the upstream status. The reachability of `FAILED` for the evidence and assessment sections is an open decision, recorded in `04_INPUT_AND_INFERENCE_PIPELINE.md` §11 and `07_INTEGRATION_TESTING.md` §19, UD-03.
3. A required-payload table exists for every section and status, and the invariants in §2.4 are enforced and tested.
4. `InputSection.status == OK` means the request passed validation, and nothing more.
5. `ComparisonSection` always has a well-defined state, and "always present" is not treated as "always meaningful".
6. The summary strip shows **two panels** and never a single combined verdict.
7. The comparison uses the **five descriptive states** and the exhaustive mapping in §8.2, covering every (classifier status and label) × (assessment status and outcome) combination.
8. No comparison sentence contains "agree", "confirm", "verified", or "prove", and the permanent note about agreement is present in every case.
9. The evidence-absence wording states that the local corpus did not return evidence meeting the retrieval criteria, and that this does not establish that relevant evidence does not exist elsewhere.
10. `rights` and `storage_scope` are distinct, and a storage statement appears **only** when the stored metadata records it.
11. Every rendered value comes from a stored record. No placeholder, no guess, and **no manufactured link** — a missing URL renders as "Source URL not available."
12. Limitations are visible at three levels and never inside a collapsed expander.
13. All eight error and fallback cases from document 00 §9.4 have a defined rendering, a degraded section stays visible, and no degraded section produces a result value.
14. The input-rejection view is a separate view, and no report model is constructed for an invalid input.
15. The three identities — registry, analysis, cache — are distinct, and the analysis fingerprint covers the input hashes and every configuration and artifact version.
16. The rerun rules hold: a display change never re-runs, new input creates a new analysis, and a configuration change invalidates the cached report.
17. Document 05 adds no second cache for the model artifacts.
18. The privacy contract is itemised and accurate, and the hosted-backend disclosure matches document 06's actual prompt payload, verified by test.
19. A clear-session control exists, with honest wording about what it does and does not clear.
20. The sample path is separate from the uploader, still passes document 04's validation, and yields to a user upload.
21. The export preserves the retrieved-versus-cited distinction, includes the rejected citations, escapes untrusted text, states its contents before the download, and **excludes the dataset-split and leakage diagnostics**.
22. The export contains exactly the enumerated fields, and a test asserts it.
23. Every untrusted string is escaped, in both the page and the export.
24. The copy is centralised, and the forbidden-wording, scope-distinctness, and manual-review layers are all in place.
25. The warning messages describe the analysis and never explain the prediction.
26. The text-rendering layer is pure and complete, and the copy, escaping, and metadata checks run against it.
27. **A manual review of representative successful and degraded reports has been performed and recorded** — the fourth layer of the copy check.
28. The fast test suite passes with no model and no network access.

---

## 21. Known limitations

| Limitation | Consequence | Status |
|---|---|---|
| The interface cannot make a small model sound better than it is | A weak explanation may read confidently | The verdict does not depend on the explanation, and the scope text is always visible |
| The comparison is a description of two outputs | It can be read as more than it is | The wording is deliberately descriptive, and the permanent note is always present |
| The corpus is small and local | Most claims yield `NO_EVIDENCE` | The wording states that this does not establish that evidence does not exist elsewhere |
| The assessment's citations are model-classified | A misclassification is possible | Document 06's eligibility rules and the evaluation report; the interface states that citation validity is not semantic support |
| The privacy statement depends on the implementation matching it | A change in the payload would make the notice wrong | The disclosure is derived from document 06's prompt assembly and verified by test |
| The session holds the report in memory | A browser crash loses it | The report is not written to disk by design, and the export is the durable path |
| Streamlit's session lifetime is not fully under the application's control | A report may outlive an explicit clear in some edge cases | The clear control removes it from the session; the wording does not over-promise |
| The manual copy review is a human step | It is not automated and can be skipped | It is an explicit acceptance criterion, so it cannot be silently skipped |
| Colour and icons are decorative | A screen reader needs the word | The outcome is always rendered as a word |

---

## 22. Interface summary

| Direction | Detail |
|---|---|
| Provides | The `ReportModel` and the rendered report, the plain-text rendering, and the Markdown export |
| Receives | The `ClassifierResult` and the `AssessmentContext` from `04_INPUT_AND_INFERENCE_PIPELINE.md`, and the `RetrievalResult` and the `Assessment` from `06_RAG_NON_RAG_EVALUATION.md` |
| Owns | The report structure and its per-section status contract, the two-signal summary, the comparison rules, the rendering and the escaping of untrusted text, the export, and the centralised copy |
| Must not | Recalculate the evidence outcome; override the evidence outcome in either direction; re-derive eligibility or re-classify a citation; present the classifier result and the evidence outcome as a single combined verdict; manufacture a link, a date, or a score that the upstream records do not contain |

The complete cross-component interface matrix, and the prohibitions that apply to every component, are specified in `07_INTEGRATION_TESTING.md` §11.

## 23. Module layout and repository actions

### 23.1 Modules

```text
src/reporting/
├── __init__.py
├── model.py          # ReportModel and the section dataclasses
├── comparison.py     # the direction mapping and the comparison states
├── text_render.py    # render_report_text, the pure text rendering
├── escaping.py       # the escaping helpers
├── export.py         # the Markdown export
└── copy.py           # every user-facing string, in one reviewed module

app.py                # the Streamlit interface, a thin wrapper over the model
```

The Streamlit layer holds no business logic. It builds the `ReportModel` from the upstream results, calls the text renderer for the copy, and displays the fields.

### 23.2 Repository actions

| Action | Detail |
|---|---|
| Populate `app.py` | The Streamlit entry point, currently empty |
| Populate `requirements.txt` | Currently empty. Needs the runtime dependencies, including Streamlit |
| `data/manifests/README.md` | Document the manifest layout, including the evidence corpus manifest |
| Reconcile `doc/` and `docs/` | Document 00 §11 uses `docs/`; the repository uses `doc/`. One name is chosen and the other removed |
| `models/README.md` | Created by document 03's repository actions |
| `CHANGELOG.md` | Created by document 03's repository actions |

---

## 24. Immediate next step

Implement `src/reporting/` against this contract, beginning with `model.py` and `comparison.py`, then `copy.py`, `escaping.py`, and `text_render.py` — all of which are testable without a model — and finally `app.py` as a thin Streamlit wrapper.

Perform the manual copy review required by acceptance criterion 27 and record the outcome in this document.

The next specification after this one is `07_INTEGRATION_TESTING.md`, the final document in the `00` sequence.
