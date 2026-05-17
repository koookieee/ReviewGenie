"""
report_generator.py — Generate a rich HTML usability report for paper authors.

Sections:
  1. Executive summary + confidence triage
  2. Per-claim cards (weakness, strength, methodology_gap, etc.)
     Each card has: reviewer said | LLM reasoning | evidence type | paper anchor
  3. Citation gap diff view (missing_citation claims)
  4. Strength inventory
  5. Rebuttal stubs for major weaknesses
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from claim_extractor import Claim, ClaimType
from text_anchor import TextAnchor
from trajectory_parser import ParsedTrajectory, SearchPaper
from trajectory_tracer import ClaimProvenance


# ---------------------------------------------------------------------------
# Color scheme (Nathan Lambert extended)
# ---------------------------------------------------------------------------

COLORS = {
    "strength":         "#22c55e",   # green
    "question":         "#3b82f6",   # blue
    "weakness":         "#f97316",   # orange
    "methodology_gap":  "#a855f7",   # purple
    "missing_citation": "#eab308",   # yellow
    "suggestion":       "#06b6d4",   # cyan
    "nit":              "#ef4444",   # red
}

COLOR_LABELS = {
    "strength":         "Strength",
    "question":         "Question",
    "weakness":         "Weakness",
    "methodology_gap":  "Methodology Gap",
    "missing_citation": "Missing Citation",
    "suggestion":       "Suggestion",
    "nit":              "Nit",
}

SEVERITY_BADGE = {
    "critical": ("background:#dc2626;color:white", "CRITICAL"),
    "major":    ("background:#ea580c;color:white", "MAJOR"),
    "minor":    ("background:#ca8a04;color:white", "MINOR"),
    "nit":      ("background:#6b7280;color:white", "NIT"),
}

CONFIDENCE_BADGE = {
    "high":   ("background:#16a34a;color:white", "High confidence"),
    "medium": ("background:#ca8a04;color:white", "Medium confidence"),
    "low":    ("background:#dc2626;color:white", "Low confidence — verify"),
}

EVIDENCE_LABEL = {
    "llm_confirmed_absent":  "✓ Reviewer confirmed absent from paper",
    "llm_confirmed_present": "✓ Reviewer verified present in paper",
    "search_found":          "🔍 Found in literature search",
    "read_section":          "📖 Derived from reading paper section",
    "reasoning_only":        "💭 Based on reasoning only (no direct verification)",
}


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text), quote=True)


def _badge(style: str, label: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:11px;font-weight:600;{style}">{_e(label)}</span>'
    )


def _section_card(claim: Claim, prov: ClaimProvenance | None, anchor: TextAnchor | None) -> str:
    color = COLORS.get(claim.claim_type, "#6b7280")
    type_label = COLOR_LABELS.get(claim.claim_type, claim.claim_type)
    sev_style, sev_label = SEVERITY_BADGE.get(claim.severity, ("background:#6b7280;color:white", claim.severity.upper()))
    conf_style = ""
    conf_label = ""
    ev_label = ""
    if prov:
        cs, cl = CONFIDENCE_BADGE.get(prov.confidence, ("background:#6b7280;color:white", prov.confidence))
        conf_style, conf_label = cs, cl
        ev_label = EVIDENCE_LABEL.get(prov.evidence_type, prov.evidence_type)

    anchor_html = ""
    if anchor and not anchor.not_found and anchor.anchor_quote:
        anchor_html = f"""
        <div style="margin-top:12px;background:#f8fafc;border-left:3px solid {color};
             padding:10px 14px;border-radius:0 6px 6px 0;">
          <div style="font-size:11px;color:#64748b;font-weight:600;
               text-transform:uppercase;margin-bottom:4px;">Paper passage this refers to</div>
          <blockquote style="margin:0;font-family:monospace;font-size:13px;
               color:#1e293b;white-space:pre-wrap;">{_e(anchor.anchor_quote)}</blockquote>
          {"<div style='margin-top:6px;font-size:12px;color:#475569;'>"+_e(anchor.anchor_context)+"</div>" if anchor.anchor_context else ""}
          {"<div style='margin-top:4px;font-size:11px;color:#94a3b8;'>≈ line " + str(anchor.anchor_line_approx) + " in source</div>" if anchor.anchor_line_approx else ""}
        </div>"""

    reasoning_html = ""
    if prov and prov.reasoning_trail:
        reasoning_html = f"""
        <div style="margin-top:12px;">
          <div style="font-size:11px;color:#64748b;font-weight:600;
               text-transform:uppercase;margin-bottom:4px;">Reviewer's reasoning process</div>
          <div style="background:#fefce8;border:1px solid #fef08a;border-radius:6px;
               padding:10px 14px;font-size:13px;color:#713f12;font-style:italic;
               white-space:pre-wrap;">{_e(prov.reasoning_trail[:600])}{"..." if len(prov.reasoning_trail) > 600 else ""}</div>
        </div>"""

    search_html = ""
    if prov and prov.supporting_search_papers:
        sp_items = "".join(
            f'<li style="margin:2px 0;"><code style="font-size:12px;">[{_e(sp.arxiv_id)}]</code> '
            f'{_e(sp.title[:80])}</li>'
            for sp in prov.supporting_search_papers[:3]
        )
        search_html = f"""
        <div style="margin-top:10px;">
          <div style="font-size:11px;color:#64748b;font-weight:600;
               text-transform:uppercase;margin-bottom:4px;">Supporting literature</div>
          <ul style="margin:0;padding-left:18px;font-size:13px;color:#334155;">{sp_items}</ul>
        </div>"""

    ev_html = ""
    if ev_label:
        ev_html = f'<div style="margin-top:8px;font-size:12px;color:#475569;">{_e(ev_label)}</div>'

    return f"""
  <div id="{_e(claim.id)}" style="border:1px solid #e2e8f0;border-radius:8px;
       margin-bottom:16px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
    <div style="background:{color}18;border-bottom:3px solid {color};
         padding:12px 16px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
      <span style="font-weight:700;color:{color};font-size:13px;">[{_e(claim.id)}]</span>
      <span style="font-weight:600;color:{color};font-size:13px;">{_e(type_label)}</span>
      {_badge(sev_style, sev_label)}
      {_badge(conf_style, conf_label) if conf_label else ""}
    </div>
    <div style="padding:14px 16px;">
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Reviewer said</div>
      <div style="font-size:14px;color:#1e293b;line-height:1.6;
           border-left:3px solid {color};padding-left:12px;">{_e(claim.review_text)}</div>
      {anchor_html}
      {ev_html}
      {reasoning_html}
      {search_html}
    </div>
  </div>"""


def _citation_gap_card(
    claim: Claim,
    prov: ClaimProvenance | None,
    anchor: TextAnchor | None,
    search_paper: SearchPaper | None,
) -> str:
    """Render the citation gap diff view card for a missing_citation claim."""
    color = COLORS["missing_citation"]

    # Left panel: paper passage where citation should appear
    if anchor and anchor.insertion_quote:
        left_content = f"""
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Your paper — where citation belongs</div>
      <blockquote style="margin:0;padding:10px 12px;background:#fefce8;
           border-left:3px solid {color};border-radius:0 6px 6px 0;
           font-family:monospace;font-size:13px;color:#1e293b;
           white-space:pre-wrap;">{_e(anchor.insertion_quote)}</blockquote>"""
    elif anchor and anchor.anchor_quote and not anchor.not_found:
        left_content = f"""
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Your paper — related passage</div>
      <blockquote style="margin:0;padding:10px 12px;background:#fefce8;
           border-left:3px solid {color};border-radius:0 6px 6px 0;
           font-family:monospace;font-size:13px;color:#1e293b;
           white-space:pre-wrap;">{_e(anchor.anchor_quote)}</blockquote>"""
    else:
        left_content = f"""
      <div style="font-size:13px;color:#64748b;font-style:italic;">
        No specific anchor found — this citation may belong in multiple places.</div>"""

    # Right panel: the missing paper
    if search_paper:
        right_content = f"""
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Missing paper</div>
      <div style="font-weight:600;color:#1e293b;font-size:13px;margin-bottom:4px;">
        {_e(search_paper.title)}</div>
      <div style="font-size:12px;color:#64748b;margin-bottom:6px;">
        arXiv:{_e(search_paper.arxiv_id)} · {search_paper.citation_count} citations</div>
      <div style="font-size:13px;color:#334155;line-height:1.5;">
        {_e((search_paper.abstract or (prov.missing_paper_abstract if prov else ""))[:400])}...</div>"""
    elif prov and prov.missing_paper_abstract:
        right_content = f"""
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Missing paper (from search)</div>
      <div style="font-size:13px;color:#334155;line-height:1.5;">
        {_e(prov.missing_paper_abstract[:400])}...</div>"""
    else:
        # Fall back to what the reviewer said
        ref_names = ", ".join(claim.cited_paper_titles[:3]) if claim.cited_paper_titles else "Unknown paper"
        right_content = f"""
      <div style="font-size:11px;color:#64748b;font-weight:600;
           text-transform:uppercase;margin-bottom:6px;">Missing paper</div>
      <div style="font-size:13px;color:#334155;">{_e(ref_names)}</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
        (Abstract not found in search results)</div>"""

    reasoning_text = prov.reasoning_trail[:400] if prov else ""

    # Rebuttal options
    rebuttal_html = """
    <div style="margin-top:14px;background:#f0f9ff;border:1px solid #bae6fd;
         border-radius:6px;padding:12px 14px;">
      <div style="font-size:11px;color:#0369a1;font-weight:600;
           text-transform:uppercase;margin-bottom:8px;">How to respond</div>
      <div style="font-size:13px;color:#0c4a6e;line-height:1.7;">
        <strong>Option A — Cite &amp; compare:</strong>
        Add a citation and brief comparison in your related work or experiments.<br>
        <strong>Option B — Scope argument:</strong>
        Explain why this paper is out of scope (different setting, later work, etc.).<br>
        <strong>Option C — Acknowledge:</strong>
        Add to limitations or future work section.
      </div>
    </div>"""

    return f"""
  <div id="gap-{_e(claim.id)}" style="border:2px solid {color};border-radius:8px;
       margin-bottom:20px;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,0.10);">
    <div style="background:{color}22;border-bottom:3px solid {color};
         padding:12px 16px;display:flex;align-items:center;gap:8px;">
      <span style="font-size:18px;">🔗</span>
      <span style="font-weight:700;color:#854d0e;font-size:14px;">
        Citation Gap [{_e(claim.id)}]</span>
      <span style="font-size:13px;color:#713f12;">{_e(", ".join(claim.cited_paper_titles[:2]) if claim.cited_paper_titles else "Unknown")}</span>
    </div>
    <div style="padding:14px 16px;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;
           margin-bottom:12px;">
        <div style="border:1px solid #e2e8f0;border-radius:6px;padding:12px;">
          {left_content}
        </div>
        <div style="border:1px solid #e2e8f0;border-radius:6px;padding:12px;">
          {right_content}
        </div>
      </div>
      {"<div style='background:#fffbeb;border:1px solid #fef08a;border-radius:6px;padding:10px 12px;font-size:13px;color:#713f12;font-style:italic;margin-bottom:10px;'>"+_e(reasoning_text[:400])+"</div>" if reasoning_text else ""}
      {rebuttal_html}
    </div>
  </div>"""


def _rebuttal_stub(claim: Claim, anchor: TextAnchor | None) -> str:
    anchor_ref = ""
    if anchor and not anchor.not_found and anchor.anchor_quote:
        anchor_ref = f' (referring to: "{anchor.anchor_quote[:80]}...")'
    return f"""
  <div style="border:1px solid #e2e8f0;border-radius:6px;margin-bottom:12px;
       padding:12px 14px;background:#f8fafc;">
    <div style="font-weight:600;color:#1e293b;font-size:13px;margin-bottom:6px;">
      [{_e(claim.id)}] {_e(claim.review_text[:120])}...</div>
    <div style="font-size:12px;color:#64748b;margin-bottom:8px;">{_e(anchor_ref)}</div>
    <div style="font-size:13px;color:#334155;line-height:1.7;
         background:#fff;border:1px solid #e2e8f0;border-radius:4px;padding:10px;">
      <strong>If incorrect:</strong> [Your counter-evidence here]<br>
      <strong>If partially correct:</strong> [Acknowledge + clarification/fix]<br>
      <strong>If valid:</strong> [Planned experiment / revision commitment]
    </div>
  </div>"""


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    def generate(
        self,
        claims: list[Claim],
        provenances: list[ClaimProvenance],
        anchors: list[TextAnchor],
        pt: ParsedTrajectory,
        paper_title: str = "",
        output_path: str | Path = "usability_report.html",
    ) -> Path:
        prov_by_id = {p.claim_id: p for p in provenances}
        anchor_by_id = {a.claim_id: a for a in anchors}
        search_by_id = {p.arxiv_id: p for p in pt.search_results}

        # Partition claims
        weaknesses = [c for c in claims if c.claim_type == "weakness"]
        strengths = [c for c in claims if c.claim_type == "strength"]
        questions = [c for c in claims if c.claim_type == "question"]
        missing_citations = [c for c in claims if c.claim_type == "missing_citation"]
        methodology_gaps = [c for c in claims if c.claim_type == "methodology_gap"]
        suggestions = [c for c in claims if c.claim_type == "suggestion"]
        nits = [c for c in claims if c.claim_type == "nit"]

        # Triage by confidence
        high_conf = [c for c in claims if prov_by_id.get(c.id) and prov_by_id[c.id].confidence == "high"]
        med_conf  = [c for c in claims if prov_by_id.get(c.id) and prov_by_id[c.id].confidence == "medium"]
        low_conf  = [c for c in claims if prov_by_id.get(c.id) and prov_by_id[c.id].confidence == "low"]

        critical_claims = [c for c in weaknesses + methodology_gaps if c.severity == "critical"]
        major_claims    = [c for c in weaknesses + methodology_gaps if c.severity == "major"]

        title_str = paper_title or pt.session_id or "Paper Review"

        # ---- Build HTML ----
        html_parts: list[str] = []
        html_parts.append(_html_head(title_str))

        # Header
        html_parts.append(f"""
<div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);color:white;
     padding:32px 40px;margin-bottom:0;">
  <h1 style="margin:0 0 8px;font-size:24px;font-weight:700;">Review Usability Report</h1>
  <div style="font-size:14px;opacity:0.85;">{_e(title_str)}</div>
  <div style="font-size:12px;opacity:0.65;margin-top:4px;">
    {len(claims)} claims extracted · Model: {_e(pt.model_name)} · Cutoff: {_e(pt.paper_cutoff)}
  </div>
</div>""")

        # ── SECTION 1: Triage panel ──
        html_parts.append("""<div style="padding:24px 40px;">""")
        html_parts.append("""<h2 style="font-size:18px;color:#1e293b;margin:0 0 16px;">
  ⚡ Confidence Triage — Where to focus first</h2>""")
        html_parts.append(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:28px;">
  <div style="background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#15803d;font-size:16px;">{len(high_conf)} claims</div>
    <div style="font-weight:600;color:#166534;margin:4px 0;">High confidence</div>
    <div style="font-size:12px;color:#166534;">Directly verified against paper or search.
    Address these first — reviewer likely correct.</div>
    <div style="margin-top:8px;font-size:12px;">
      {" ".join(f'<a href="#{_e(c.id)}" style="color:#15803d;text-decoration:none;background:#bbf7d0;padding:1px 5px;border-radius:3px;margin:1px;">{_e(c.id)}</a>' for c in high_conf[:12])}
    </div>
  </div>
  <div style="background:#fefce8;border:1px solid #fde047;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#b45309;font-size:16px;">{len(med_conf)} claims</div>
    <div style="font-weight:600;color:#92400e;margin:4px 0;">Medium confidence</div>
    <div style="font-size:12px;color:#92400e;">Plausible but not fully verified.
    Review carefully — may be correct or off-base.</div>
    <div style="margin-top:8px;font-size:12px;">
      {" ".join(f'<a href="#{_e(c.id)}" style="color:#b45309;text-decoration:none;background:#fef9c3;padding:1px 5px;border-radius:3px;margin:1px;">{_e(c.id)}</a>' for c in med_conf[:12])}
    </div>
  </div>
  <div style="background:#fee2e2;border:1px solid #fca5a5;border-radius:8px;padding:16px;">
    <div style="font-weight:700;color:#dc2626;font-size:16px;">{len(low_conf)} claims</div>
    <div style="font-weight:600;color:#991b1b;margin:4px 0;">Low confidence</div>
    <div style="font-size:12px;color:#991b1b;">Based on LLM reasoning only.
    These may be wrong — verify before acting on them.</div>
    <div style="margin-top:8px;font-size:12px;">
      {" ".join(f'<a href="#{_e(c.id)}" style="color:#dc2626;text-decoration:none;background:#fee2e2;padding:1px 5px;border-radius:3px;margin:1px;">{_e(c.id)}</a>' for c in low_conf[:12])}
    </div>
  </div>
</div>""")

        # ── SECTION 2: Weaknesses ──
        if weaknesses or methodology_gaps:
            html_parts.append(_section_header("Weaknesses & Methodology Gaps",
                                              f"{len(weaknesses)+len(methodology_gaps)} issues"))
            for c in (weaknesses + methodology_gaps):
                html_parts.append(_section_card(c, prov_by_id.get(c.id), anchor_by_id.get(c.id)))

        # ── SECTION 3: Citation Gaps ──
        if missing_citations:
            html_parts.append(_section_header("Citation Gap Analysis",
                                              f"{len(missing_citations)} missing references"))
            for c in missing_citations:
                prov = prov_by_id.get(c.id)
                anchor = anchor_by_id.get(c.id)
                # Find the search paper for this missing citation
                sp = None
                for aid in c.cited_papers:
                    if aid in search_by_id:
                        sp = search_by_id[aid]
                        break
                html_parts.append(_citation_gap_card(c, prov, anchor, sp))

        # ── SECTION 4: Strengths ──
        if strengths:
            html_parts.append(_section_header("Strengths", f"{len(strengths)} positives identified"))
            for c in strengths:
                html_parts.append(_section_card(c, prov_by_id.get(c.id), anchor_by_id.get(c.id)))

        # ── SECTION 5: Questions ──
        if questions:
            html_parts.append(_section_header("Reviewer Questions",
                                              "Answer these in your rebuttal"))
            for c in questions:
                html_parts.append(_section_card(c, prov_by_id.get(c.id), anchor_by_id.get(c.id)))

        # ── SECTION 6: Suggestions ──
        if suggestions:
            html_parts.append(_section_header("Suggestions for Improvement",
                                              f"{len(suggestions)} actionable items"))
            for c in suggestions:
                html_parts.append(_section_card(c, prov_by_id.get(c.id), anchor_by_id.get(c.id)))

        # ── SECTION 7: Nits ──
        if nits:
            html_parts.append(_section_header("Minor / Presentation Nits",
                                              "Low priority — fix in camera-ready"))
            for c in nits:
                html_parts.append(_section_card(c, prov_by_id.get(c.id), anchor_by_id.get(c.id)))

        # ── SECTION 8: Rebuttal stubs ──
        rebuttable = [c for c in critical_claims + major_claims
                      if c.claim_type in ("weakness", "methodology_gap")]
        if rebuttable:
            html_parts.append(_section_header("Rebuttal Stubs",
                                              "Templates for major weaknesses"))
            html_parts.append("""<p style="font-size:13px;color:#64748b;margin-bottom:16px;">
  Fill in the blanks — these are starting points for your rebuttal response.</p>""")
            for c in rebuttable:
                html_parts.append(_rebuttal_stub(c, anchor_by_id.get(c.id)))

        html_parts.append("</div>")  # close main padding div
        html_parts.append(_html_foot())

        out = Path(output_path)
        out.write_text("".join(html_parts), encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# HTML boilerplate
# ---------------------------------------------------------------------------

def _section_header(title: str, subtitle: str = "") -> str:
    return f"""
<h2 style="font-size:18px;color:#1e293b;margin:28px 0 12px;padding-bottom:8px;
     border-bottom:2px solid #e2e8f0;">{_e(title)}
  {"<span style='font-size:13px;font-weight:400;color:#64748b;margin-left:10px;'>"+_e(subtitle)+"</span>" if subtitle else ""}
</h2>"""


def _html_head(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Review Report — {_e(title)}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 0; background: #f8fafc; color: #1e293b; line-height: 1.5; }}
  a {{ color: #2563eb; }}
  code {{ background: #f1f5f9; padding: 1px 4px; border-radius: 3px;
          font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; }}
  blockquote {{ margin: 0; }}
  @media (max-width: 768px) {{
    div[style*="grid-template-columns:1fr 1fr"] {{
      grid-template-columns: 1fr !important;
    }}
  }}
</style>
</head>
<body>
"""


def _html_foot() -> str:
    return """
<div style="background:#f1f5f9;border-top:1px solid #e2e8f0;padding:20px 40px;
     font-size:12px;color:#94a3b8;text-align:center;">
  Generated by ReviewGenie Annotation Pipeline
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    from trajectory_parser import parse_trajectory
    from claim_extractor import ClaimExtractor
    from trajectory_tracer import TrajectoryTracer
    from text_anchor import TextAnchorEngine

    api_key = os.environ.get("GEMINI_API_KEY", "")
    traj_path = sys.argv[1] if len(sys.argv) > 1 else "trajectory_1612.00472_fab1.0_new.json"
    paper_path = sys.argv[2] if len(sys.argv) > 2 else "main.tex"

    pt = parse_trajectory(traj_path)
    paper_text = open(paper_path).read()
    extractor = ClaimExtractor(api_key=api_key)
    claims = extractor.extract(pt.final_review)
    tracer = TrajectoryTracer(api_key=api_key)
    provenances = tracer.trace(claims, pt)
    engine = TextAnchorEngine(api_key=api_key)
    anchors = engine.anchor(claims, provenances, paper_text)

    gen = ReportGenerator()
    out = gen.generate(claims, provenances, anchors, pt,
                       paper_title="Understanding image motion with group representations",
                       output_path="usability_report.html")
    print(f"Report written to: {out}")
