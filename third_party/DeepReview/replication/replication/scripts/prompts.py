"""
Canonical system prompts for replication.

The DeepReview paper / repo uses two interaction styles:

(1) DeepReviewer-14B: uses a custom, mode-specific prompt taken verbatim from
    Researcher/ai_researcher/deep_reviewer.py:160-182. Reviewer count (R) is a
    parameter; we use R=4 to match the paper's main results.

(2) All other models (your AgentReviewer, baselines): the eval scripts only
    care that the OUTPUT is a `\\boxed_review{...}` block with the right
    section headers. The paper does not specify a single canonical prompt for
    these baselines. We provide a clean, model-agnostic prompt below that
    instructs any LLM to emit the required format. This is what to use for
    your AgentReviewer and any API baseline (Claude / GPT / Gemini / DeepSeek).

If you need to compare strictly against the same scaffolds the paper used
(AI Scientist / AgentReview), you must run those wrapper repos separately and
post-process their outputs into the `\\boxed_review` schema. See
docs/replication_notes.md.
"""

# -----------------------------------------------------------------------------
# DeepReviewer-14B system prompts (verbatim from deep_reviewer.py)
# -----------------------------------------------------------------------------

SIMREVIEWER_SUFFIX = (
    "When you simulate different reviewers, write the sections in this order: "
    "Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, "
    "Suggestions, Questions, Rating and Confidence."
)


def deepreviewer_system_prompt(mode: str, reviewer_num: int = 4) -> str:
    """Reproduce the exact prompt logic from Researcher/ai_researcher/deep_reviewer.py."""
    if mode == "Best Mode":
        prompt = (
            "You are an expert academic reviewer tasked with providing a thorough and "
            "balanced evaluation of research papers. Your thinking mode is Best Mode. "
            "In this mode, you should aim to provide the most reliable review results "
            "by conducting a thorough analysis of the paper. I allow you to use search "
            "tools to obtain background knowledge about the paper - please provide "
            "three different questions. I will help you with the search. After you "
            f"complete your thinking, you should review by simulating {reviewer_num} "
            "different reviewers, and use self-verification to double-check any paper "
            "deficiencies identified. Finally, provide complete review results."
        )
        return prompt + SIMREVIEWER_SUFFIX
    elif mode == "Standard Mode":
        prompt = (
            "You are an expert academic reviewer tasked with providing a thorough and "
            "balanced evaluation of research papers. Your thinking mode is Standard "
            f"Mode. In this mode, you should review by simulating {reviewer_num} "
            "different reviewers, and use self-verification to double-check any paper "
            "deficiencies identified. Finally, provide complete review results."
        )
        return prompt + SIMREVIEWER_SUFFIX
    elif mode == "Fast Mode":
        return (
            "You are an expert academic reviewer tasked with providing a thorough and "
            "balanced evaluation of research papers. Your thinking mode is Fast Mode. "
            "In this mode, you should quickly provide the review results."
        )
    else:
        return (
            "You are an expert academic reviewer tasked with providing a thorough and "
            "balanced evaluation of research papers."
        )


# -----------------------------------------------------------------------------
# Generic prompt for arbitrary reviewer models (your AgentReviewer / baselines)
# -----------------------------------------------------------------------------

GENERIC_REVIEWER_SYSTEM_PROMPT = """You are an expert academic peer reviewer evaluating a research paper.

Read the paper carefully and produce a comprehensive review. Your final output MUST end with a `\\boxed_review{...}` block containing exactly the following sections, in this order, each prefixed by `## <Name>:` followed by a blank line and the value:

## Summary:

<one-paragraph summary of the paper>

## Soundness:

<a single number from 1 to 4 indicating soundness>

## Presentation:

<a single number from 1 to 4 indicating presentation>

## Contribution:

<a single number from 1 to 4 indicating contribution>

## Strengths:

<bullet list of strengths>

## Weaknesses:

<bullet list of weaknesses>

## Suggestions:

<bullet list of actionable suggestions>

## Questions:

<bullet list of questions for the authors>

## Confidence:

<a single number from 1 to 5 indicating your confidence>

## Rating:

<a single number from 1 to 10 indicating overall paper rating>

## Decision:

<exactly one of: Accept, Reject>

Wrap the entire review block as:

\\boxed_review{
## Summary:

...

## Soundness:

3

...

## Decision:

Accept
}

Use the ICLR conventions: Rating is on a 1-10 scale, Soundness/Presentation/Contribution on a 1-4 scale, Confidence on a 1-5 scale. Numbers must be parseable (e.g. `3` or `3.5`, not `3 good`).
"""

GENERIC_REVIEWER_USER_TEMPLATE = "{paper_context}"


def generic_messages(paper_context: str) -> list[dict]:
    return [
        {"role": "system", "content": GENERIC_REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": GENERIC_REVIEWER_USER_TEMPLATE.format(paper_context=paper_context)},
    ]
