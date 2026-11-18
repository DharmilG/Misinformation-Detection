"""Minimal localhost experiment instrument for Study 3.

Binds only to 127.0.0.1, uses the Python standard library, and writes anonymous
participant responses to JSONL.  It is a research instrument, not a hosted
service, and does not expose the classifier or any model weights.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


CONDITIONS = ("C0", "C1", "C2", "C3")
DECISIONS = ("MATCHED", "MISMATCHED")


class StudyError(ValueError):
    """Raised for an invalid experiment configuration."""


@dataclass
class Experiment:
    cases: list[dict[str, Any]]
    output: Path
    host: str = "127.0.0.1"
    port: int = 8765
    _answers: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.host != "127.0.0.1":
            raise StudyError("the study server must bind to 127.0.0.1 only")
        if not self.cases:
            raise StudyError("no cases supplied")
        for case in self.cases:
            for key in ("id", "image", "caption", "gold_label"):
                if key not in case:
                    raise StudyError(f"case missing {key}: {case}")
            if case["gold_label"] not in DECISIONS:
                raise StudyError(f"bad gold_label: {case['gold_label']!r}")

    def condition_order(self, participant_index: int) -> tuple[str, ...]:
        """Latin-square rotation so each condition is seen in each position."""
        offset = participant_index % len(CONDITIONS)
        return CONDITIONS[offset:] + CONDITIONS[:offset]

    def schedule(self, participant_index: int) -> list[dict[str, Any]]:
        """One case per condition, in counterbalanced order."""
        order = self.condition_order(participant_index)
        cases = self.cases[: len(order)]
        return [dict(case, condition=cond) for case, cond in zip(cases, order)]

    def record(
        self,
        *,
        participant: str,
        case_id: str,
        condition: str,
        decision: str,
        confidence: int,
        elapsed_ms: int,
        evidence_viewed: bool,
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise StudyError(f"bad decision: {decision!r}")
        if not 1 <= confidence <= 7:
            raise StudyError("confidence must be in 1..7")
        row = {
            "participant": participant,
            "case_id": case_id,
            "condition": condition,
            "decision": decision,
            "confidence": confidence,
            "elapsed_ms": elapsed_ms,
            "evidence_viewed": evidence_viewed,
        }
        self._answers.append(row)
        return row

    def flush(self) -> Path:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self._answers:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.output

    def metrics(self) -> dict[str, Any]:
        by_condition: dict[str, dict[str, Any]] = {}
        gold = {str(c["id"]): c["gold_label"] for c in self.cases}
        for condition in CONDITIONS:
            rows = [r for r in self._answers if r["condition"] == condition]
            if not rows:
                by_condition[condition] = {"n": 0}
                continue
            correct = sum(gold.get(r["case_id"]) == r["decision"] for r in rows)
            by_condition[condition] = {
                "n": len(rows),
                "accuracy": round(correct / len(rows), 6),
                "mean_time_ms": round(
                    sum(r["elapsed_ms"] for r in rows) / len(rows), 3
                ),
                "mean_confidence": round(
                    sum(r["confidence"] for r in rows) / len(rows), 6
                ),
            }
        return {
            "answers": len(self._answers),
            "participants": len({r["participant"] for r in self._answers}),
            "by_condition": by_condition,
            "automation_bias_rate": "[PLANNED - NOT YET MEASURED]",
            "interpretation": "[PLANNED - NOT YET MEASURED]",
        }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _page(experiment: Experiment, participant: str, index: int) -> bytes:
    schedule = experiment.schedule(int(participant) - 1)
    case = schedule[index]
    condition = case["condition"]
    info = {
        "C0": "No model output is shown.",
        "C1": "Model verdict and confidence are shown.",
        "C2": "Model output and retrieved evidence are shown.",
        "C3": "Full trace, evidence and abstention flag are shown.",
    }[condition]
    options = "".join(
        f'<label><input type="radio" name="decision" value="{d}" required> {d}</label> '
        for d in DECISIONS
    )
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Verification study {html.escape(participant)}</title></head><body>
<h1>Image–caption verification</h1>
<p>Participant {html.escape(participant)} — case {index + 1} of {len(schedule)} —
condition {condition}</p>
<p>{html.escape(info)}</p>
<img src="/image?path={html.escape(str(case['image']))}" alt="case image" width="400">
<p><strong>Caption:</strong> {html.escape(str(case['caption']))}</p>
<form method="post" action="/answer">
<input type="hidden" name="participant" value="{html.escape(participant)}">
<input type="hidden" name="case_id" value="{html.escape(str(case['id']))}">
<input type="hidden" name="condition" value="{condition}">
<input type="hidden" name="index" value="{index}">
<input type="hidden" name="started" value="{int(time.time() * 1000)}">
<fieldset><legend>Your decision</legend>{options}</fieldset>
<p><label>Confidence (1=guess, 7=certain)
<input type="number" name="confidence" min="1" max="7" required></label></p>
<p><label><input type="checkbox" name="evidence" value="1"> I looked at evidence</label></p>
<button type="submit">Submit</button></form></body></html>"""
    return body.encode("utf-8")


def make_handler(experiment: Experiment) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return  # no participant-identifying logs

        def _send(self, body: bytes, status: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/image":
                raw = (query.get("path") or [""])[0]
                path = Path(raw).resolve()
                if not path.is_file() or not str(path).startswith(str(Path.cwd().resolve())):
                    self._send(b"not found", 404)
                    return
                self._send(path.read_bytes(), ctype="image/jpeg")
                return
            participant = (query.get("participant") or ["1"])[0]
            index = int((query.get("index") or ["0"])[0])
            self._send(_page(experiment, participant, index))

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/answer":
                self._send(b"not found", 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            fields = parse_qs(self.rfile.read(length).decode("utf-8"))
            value = lambda key: (fields.get(key) or [""])[0]
            participant = value("participant")
            index = int(value("index"))
            experiment.record(
                participant=participant,
                case_id=value("case_id"),
                condition=value("condition"),
                decision=value("decision"),
                confidence=int(value("confidence")),
                elapsed_ms=int(time.time() * 1000) - int(value("started")),
                evidence_viewed=value("evidence") == "1",
            )
            experiment.flush()
            next_index = index + 1
            if next_index >= len(experiment.schedule(int(participant) - 1)):
                self._send(
                    b"<html><body><h1>Done</h1><p>Thank you.</p></body></html>"
                )
                return
            self.send_response(303)
            self.send_header(
                "Location", f"/?participant={participant}&index={next_index}"
            )
            self.end_headers()

    return Handler


def serve(experiment: Experiment) -> None:
    """Start the local study server until interrupted."""
    server = ThreadingHTTPServer((experiment.host, experiment.port), make_handler(experiment))
    print(f"Study server on http://{experiment.host}:{experiment.port}/?participant=1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        experiment.flush()
        server.server_close()
