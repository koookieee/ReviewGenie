Paper Reviewer — RL Training Pipeline
This is a reinforcement learning system that trains a model (Qwen3) to write academic paper reviews, using SkyRL + Harbor for agentic training.

High-Level Flow
Dataset prep (prepare_paper_reviewer_dataset.py) — Downloads papers from arXiv (LaTeX source + metadata), creates task directories with instruction prompts, Dockerfiles, and config for Harbor.

Agent execution — For each paper, a Claude Code agent runs inside an E2B sandbox and performs a 6-phase review (defined in paper_reviewer_instruction_template.md):

Read the paper → Deep literature search → Novelty assessment → Impact analysis → Methodology critique → Framing
Reward (paper_reviewer_generator.py) — An LLM judge (Gemini 3 Flash) scores the review on 5 criteria (comprehension, substance, insight, issue overlap with human reviews, calibration) using the template in llm_judge_instruction.md.

Training — SkyRL uses GRPO to update the policy model based on collected trajectories.

Infrastructure Components
Component	File	Purpose
Search API	search_api.py	HTTP server over arxiv-search-kit (928K papers, semantic + BM25 search, citations, LaTeX download)
Stream Proxy	stream_proxy.py	Translates Anthropic API ↔ OpenAI API so Claude Code can talk to vLLM
Training entry	entrypoints/main_paper_reviewer.py	Launches SkyRL with Ray + FSDP across GPUs
Launch script	start_training.sh	Orchestrates everything (proxy, Ray, vLLM, training)
Key Design Decisions
No automated verifier — review quality can't be programmatically checked, so an LLM judge provides the reward signal
Claude Code session JSONL is parsed directly to reconstruct chat history (Harbor doesn't populate all_messages metadata)
SkyRL patch (skyrl_tool_calling.patch) fixes a vLLM bug where tool_choice: "auto" gets rejected
