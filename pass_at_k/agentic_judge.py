"""agentic_judge.py — three-call judge for paper review scoring.

Call 1 — Overlap (human reviews + AI review only):
  No paper in context — judge cannot invent points beyond what humans said.

Call 2 — Fabrication (paper body + AI review only):
  No human reviews — judge checks AI claims purely against the paper.

Call 3 — Rest (paper body + human reviews + AI review):
  Scores comprehension, substance, insight, calibration.

All three calls run in parallel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _criterion_schema(score_type: genai_types.Type, list_field: str | None = None,
                      list_item_props: dict | None = None) -> genai_types.Schema:
    props: dict = {
        "justification": genai_types.Schema(type=genai_types.Type.STRING),
        "evidence": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
        "score": genai_types.Schema(type=score_type),
    }
    required = ["justification", "score"]
    if list_field and list_item_props:
        props[list_field] = genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties=list_item_props,
            ),
        )
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties=props,
        required=required,
    )


# ---------------------------------------------------------------------------
# Call 1 schema: overlap only
# ---------------------------------------------------------------------------

_OVERLAP_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "reference_verdict": genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "overall_mean": genai_types.Schema(type=genai_types.Type.NUMBER),
                "decision_consensus": genai_types.Schema(type=genai_types.Type.STRING),
                "soundness_mean": genai_types.Schema(type=genai_types.Type.NUMBER, nullable=True),
            },
        ),
        "issue_overlap": _criterion_schema(
            genai_types.Type.NUMBER,
            list_field="overlap_points",
            list_item_props={
                "point": genai_types.Schema(type=genai_types.Type.STRING),
                "raised_by": genai_types.Schema(
                    type=genai_types.Type.ARRAY,
                    items=genai_types.Schema(type=genai_types.Type.INTEGER),
                ),
                "covered_by_model": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                "evidence": genai_types.Schema(type=genai_types.Type.STRING),
            },
        ),
    },
    required=["reference_verdict", "issue_overlap"],
)

# ---------------------------------------------------------------------------
# Call 2 schema: fabrication only
# ---------------------------------------------------------------------------

_FABRICATION_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "fabrication": _criterion_schema(
            genai_types.Type.NUMBER,
            list_field="fabrication_checks",
            list_item_props={
                "claim": genai_types.Schema(type=genai_types.Type.STRING),
                "status": genai_types.Schema(type=genai_types.Type.STRING),
                "note": genai_types.Schema(type=genai_types.Type.STRING),
            },
        ),
    },
    required=["fabrication"],
)

# ---------------------------------------------------------------------------
# Call 3 schema: comprehension, substance, insight, calibration
# ---------------------------------------------------------------------------

_REST_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "comprehension": _criterion_schema(genai_types.Type.NUMBER),
        "substance_and_specificity": _criterion_schema(genai_types.Type.NUMBER),
        "insight": _criterion_schema(
            genai_types.Type.NUMBER,
            list_field="insight_observations",
            list_item_props={
                "observation": genai_types.Schema(type=genai_types.Type.STRING),
                "grounds_in": genai_types.Schema(type=genai_types.Type.STRING),
                "evidence": genai_types.Schema(type=genai_types.Type.STRING),
            },
        ),
        "calibration_pairwise": _criterion_schema(
            genai_types.Type.NUMBER,
            list_field="calibration_judgments",
            list_item_props={
                "dimension": genai_types.Schema(type=genai_types.Type.STRING),
                "verdict": genai_types.Schema(type=genai_types.Type.STRING),
                "score": genai_types.Schema(type=genai_types.Type.NUMBER),
                "evidence": genai_types.Schema(type=genai_types.Type.STRING),
            },
        ),
    },
    required=["comprehension", "substance_and_specificity", "insight", "calibration_pairwise"],
)


# ---------------------------------------------------------------------------
# Telemetry / config
# ---------------------------------------------------------------------------

@dataclass
class JudgeTelemetry:
    wall_clock_s: float = 0.0
    finish_reason_overlap: str = ""
    finish_reason_fabrication: str = ""
    finish_reason_rest: str = ""
    parse_error: str = ""


@dataclass
class AgenticJudgeConfig:
    model: str = "gemini-3.1-pro-preview"
    max_tokens_per_call: int = 32768
    temperature: float = 0.0
    # kept for backwards compat
    max_steps: int = 40
    max_wall_clock_s: float = 480.0
    final_max_tokens: int = 32768


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _overlap_prompt(human_reviews_text: str, model_review: str) -> str:
    return f"""You are an experienced area chair at a top-tier ML conference.

Score how well the model's review covers the points the human reviewers explicitly raised.

## Human Reviews

{human_reviews_text}

## Model's Review

{model_review}

---

### Issue Overlap

Extract every substantive point from the human reviews (strengths, weaknesses, questions).
Check if the model's review covers each point (by substance, not exact wording).

**CRITICAL: Only list points the human reviewers explicitly stated. Do NOT add points from your own assessment or knowledge. Every point in your list must be directly traceable to the human review text above.**

Convergent points (raised by ≥ 2 reviewers) weight 2× single-reviewer points.
Score = (weighted covered) / (total weighted).

Also extract:
- reference_verdict.overall_mean: mean of human rating scores
- reference_verdict.decision_consensus: "Accept" if >50% vote Accept, else "Reject"
- reference_verdict.soundness_mean: mean soundness if reported, else null

Return JSON. No preamble."""


def _fabrication_prompt(paper_body: str, model_review: str) -> str:
    return f"""You are an experienced area chair at a top-tier ML conference.

Check whether the model's review fabricates facts about the paper.

## Full Paper Body (up to References)

{paper_body}

## Model's Review

{model_review}

---

### Fabrication Check

Enumerate every specific factual claim in the model review: numbers, named baselines, algorithm details, section references, quoted phrases.

For each claim, scan the paper body above to verify it. The paper is the only source of truth.

Numbers may appear as prose (`0.5`), LaTeX (`\\( 0.5 \\)`), or HTML table cells (`<td>14.71%</td>`).
Names may appear in running text, reference lists, or abbreviations.

Mark each claim:
- **verified** — found in paper body. Quote matching text in note.
- **unverified** — paper positively contradicts the claim (different number present, paper states opposite). Quote the contradiction in note.
- **unverifiable** — claim is about a figure/table lost in PDF conversion. No penalty.

Do NOT mark unverified unless the paper actively contradicts the claim. Absence on first scan ≠ contradiction — check LaTeX and HTML forms too.

Score: 1.0 if zero unverified, 0.5 if one, 0.0 if ≥ 2.

Return JSON. No preamble."""


def _rest_prompt(title: str, abstract: str, paper_body: str,
                 human_reviews_text: str, model_review: str) -> str:
    return f"""You are an experienced area chair at a top-tier ML conference.

Score the model's review on: Comprehension, Substance, Insight, and Calibration.

## Paper

**Title:** {title}

**Abstract:** {abstract}

**Full Paper Body (up to References):**
{paper_body}

## Human Reviews

{human_reviews_text}

## Model's Review

{model_review}

---

### Criterion 1: Comprehension — binary {{0, 1}}  (weight 0.05)

Score 1 if the review correctly identifies the paper's core contribution, method, and claims with evidence traceable to the paper body.
Score 0 if it mischaracterizes the paper or engages only with the abstract.

### Criterion 2: Substance — binary {{0, 1}}  (weight 0.05)

Score 1 if the review raises ≥ 2 technical points that are specific, non-trivial (not from abstract), and actionable.
Score 0 otherwise.

### Criterion 3: Insight — continuous {{0.0, 0.5, 1.0}}  (weight 0.15)

Count grounded, non-obvious observations. An observation counts only if:
1. Goes beyond the abstract.
2. Grounded in something identifiable in the paper body — verify by scanning above.
3. Non-generic.

For each credited observation output: observation, grounds_in (section/element), evidence (short quote).
Score: 1.0 if ≥ 3, 0.5 if exactly 2, 0.0 if ≤ 1.

### Criterion 4: Pairwise Calibration — continuous [0.0, 1.0]  (weight 0.25)

Score two dimensions and take the mean:

**Dimension A — Decision Alignment** (binary: 0 or 1):
- Extract the model's final verdict (Accept/Reject) from its review.
- Extract the human consensus verdict from the human reviews above.
- Score 1.0 if they match, 0.0 if they do not.

**Dimension B — Internal Consistency** (continuous: 0.0, 0.5, 1.0):
- Read the model's Strengths, Weaknesses, and Overall score together.
- Score 1.0 if the verdict and score are fully consistent with the written critique: a paper praised with few weaknesses gets a high score and Accept; a paper heavily criticised gets a low score and Reject.
- Score 0.5 if there is a minor mismatch (e.g. mostly positive but slightly low score, or one unexplained inconsistency).
- Score 0.0 if there is a clear contradiction: the review raises severe weaknesses but accepts the paper, or praises it strongly but rejects it.

Final calibration score = mean(Dimension A, Dimension B).

Return JSON. No preamble."""


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class AgenticJudge:
    """Three parallel calls: overlap, fabrication, rest."""

    def __init__(self, *, api_key: str, prompt_template: str,
                 config: AgenticJudgeConfig | None = None):
        if not api_key:
            raise ValueError("AgenticJudge requires a non-empty api_key")
        # prompt_template kept for API compat but not used — prompts are built inline
        self._config = config or AgenticJudgeConfig()
        self._client = genai.Client(api_key=api_key)

    async def _call(self, prompt: str, schema: genai_types.Schema) -> tuple[dict | None, str, str]:
        contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
        try:
            resp = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._config.model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=self._config.max_tokens_per_call,
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_level=genai_types.ThinkingLevel.MEDIUM,
                    ),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            cand = (resp.candidates or [None])[0]
            finish = str(getattr(cand, "finish_reason", "") or "")
            text = ""
            if cand and cand.content:
                for part in (cand.content.parts or []):
                    if getattr(part, "text", None):
                        text += part.text
            parsed, err = _parse_judge_json(text)
            return parsed, finish, err, text
        except Exception as e:
            return None, "", f"{type(e).__name__}: {e}", ""

    async def score(self, *, title: str, abstract: str, paper_body: str,
                    human_reviews_text: str, model_review: str) -> dict:
        tel = JudgeTelemetry()
        t0 = time.monotonic()

        (overlap_r, fin_ov, err_ov, raw_ov), (fab_r, fin_fab, err_fab, raw_fab), (rest_r, fin_rest, err_rest, raw_rest) = \
            await asyncio.gather(
                self._call(_overlap_prompt(human_reviews_text, model_review), _OVERLAP_SCHEMA),
                self._call(_fabrication_prompt(paper_body, model_review), _FABRICATION_SCHEMA),
                self._call(_rest_prompt(title, abstract, paper_body, human_reviews_text, model_review), _REST_SCHEMA),
            )

        tel.wall_clock_s = time.monotonic() - t0
        tel.finish_reason_overlap = fin_ov
        tel.finish_reason_fabrication = fin_fab
        tel.finish_reason_rest = fin_rest

        if overlap_r is None or fab_r is None or rest_r is None:
            err = err_ov or err_fab or err_rest
            tel.parse_error = err
            logger.warning(f"Judge call failed: ov={err_ov} fab={err_fab} rest={err_rest}")
            tel_dict = _telemetry_to_dict(tel)
            if err_ov: tel_dict["raw_overlap"] = raw_ov
            if err_fab: tel_dict["raw_fabrication"] = raw_fab
            if err_rest: tel_dict["raw_rest"] = raw_rest
            return {"error": err, "reward": 0.0, "_judge_telemetry": tel_dict}

        scores: dict = {}
        scores["reference_verdict"] = overlap_r.get("reference_verdict", {})
        scores["issue_overlap"] = overlap_r.get("issue_overlap", {})
        scores["fabrication"] = fab_r.get("fabrication", {})
        scores["comprehension"] = rest_r.get("comprehension", {})
        scores["substance_and_specificity"] = rest_r.get("substance_and_specificity", {})
        scores["insight"] = rest_r.get("insight", {})
        scores["calibration_pairwise"] = rest_r.get("calibration_pairwise", {})
        scores["_judge_telemetry"] = _telemetry_to_dict(tel)
        return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_judge_json(text: str) -> tuple[dict | None, str]:
    if not text:
        return None, "empty_judge_output"
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None, "no_json_in_judge_output"
    try:
        return json.loads(m.group()), ""
    except json.JSONDecodeError as e:
        return None, f"json_parse: {e}"


def _telemetry_to_dict(tel: JudgeTelemetry) -> dict:
    return {
        "wall_clock_s": round(tel.wall_clock_s, 2),
        "finish_reason_overlap": tel.finish_reason_overlap,
        "finish_reason_fabrication": tel.finish_reason_fabrication,
        "finish_reason_rest": tel.finish_reason_rest,
        "parse_error": tel.parse_error,
    }