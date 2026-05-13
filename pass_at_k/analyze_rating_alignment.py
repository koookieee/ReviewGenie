import json, pathlib, re

def extract_ai_overall(review_text):
    for pat in [
        r"Overall[:\*\s]+(\d+)\s*/\s*10",
        r"\*\*Overall\*\*[:\s]+(\d+)",
        r"Overall Rating[:\s]+(\d+)",
    ]:
        m = re.search(pat, review_text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return v
    return None

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    d1 = sum((x-mx)**2 for x in xs)**0.5
    d2 = sum((y-my)**2 for y in ys)**0.5
    return num/(d1*d2) if d1*d2 else 0

def spearman(xs, ys):
    n = len(xs)
    ri = sorted(range(n), key=lambda i: xs[i])
    rj = sorted(range(n), key=lambda i: ys[i])
    rx, ry = [0]*n, [0]*n
    for rank, i in enumerate(ri): rx[i] = rank
    for rank, i in enumerate(rj): ry[i] = rank
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1 - 6*d2/(n*(n**2-1))

def kappa(ys1, ys2):
    n = len(ys1)
    po = sum(1 for a,b in zip(ys1,ys2) if a==b)/n
    ph = (sum(ys1)/n)*(sum(ys2)/n) + (1-sum(ys1)/n)*(1-sum(ys2)/n)
    return (po-ph)/(1-ph) if ph != 1 else 0

def mae(xs, ys):
    return sum(abs(x-y) for x,y in zip(xs,ys))/len(xs)

def print_stats(name, pairs):
    if not pairs:
        print(f"\n{name}: no pairs found")
        return
    n = len(pairs)
    humans = [p["human"] for p in pairs]
    ais = [p["ai"] for p in pairs]
    h_acc = [1 if h >= 6 else 0 for h in humans]
    a_acc = [1 if a >= 6 else 0 for a in ais]
    print(f"\n{name} (n={n})")
    print(f"  Pearson r         = {pearson(humans, ais):.3f}")
    print(f"  Spearman rho      = {spearman(humans, ais):.3f}")
    print(f"  MAE               = {mae(humans, ais):.3f}")
    print(f"  Cohen's kappa     = {kappa(h_acc, a_acc):.3f}  (threshold >= 6/10)")
    print(f"  Human accept rate = {sum(h_acc)/n:.1%}   AI accept rate = {sum(a_acc)/n:.1%}")

# ---- Agentic runs ----
review_markers = ("### Summary", "### Strengths", "### Scores", "**Scores**", "**Overall**")

def get_agentic_pairs(results_dir, trials_dir):
    pairs = []
    for f in sorted(pathlib.Path(results_dir).rglob("result.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("status") != "success" or not d.get("reward"): continue
            rv = (d.get("scores") or {}).get("reference_verdict", {})
            human_mean = rv.get("overall_mean")
            if not human_mean: continue
            paper_id = d["paper_id"]
            trial_name = d.get("trial_name", "")
            review = ""
            for t in pathlib.Path(trials_dir).rglob("trajectory.json"):
                if paper_id in str(t) and trial_name in str(t):
                    steps = json.loads(t.read_text()).get("steps", [])
                    msgs = [s["message"] for s in steps if s.get("source")=="agent" and s.get("message") and len(s["message"])>200]
                    review = next((m for m in reversed(msgs) if any(mk in m for mk in review_markers)), msgs[-1] if msgs else "")
                    break
            ai_score = extract_ai_overall(review)
            if ai_score is None: continue
            pairs.append({"human": float(human_mean), "ai": float(ai_score)})
        except: pass
    return pairs

# ---- Stanford ----
def get_stanford_pairs():
    pairs = []
    reviews_dir = pathlib.Path("/root/Stanford_Reviewer/reviews_proxy")
    data_dir = pathlib.Path("/root/data/pass_at_k_reviewed")
    for f in sorted(reviews_dir.glob("*_review.json")):
        try:
            r = json.loads(f.read_text())
            ai_score = r.get("numerical_score")
            if not ai_score: continue
            paper_id = f.stem.replace("_review", "")
            meta = json.loads((data_dir / paper_id / "task_metadata.json").read_text())
            hrs = meta.get("human_reviews", [])
            if not hrs: continue
            ratings = []
            for hr in hrs:
                m = re.search(r"Rating:\s*(\d+)", str(hr))
                if m: ratings.append(int(m.group(1)))
            if not ratings: continue
            human_mean = sum(ratings)/len(ratings)
            pairs.append({"human": human_mean, "ai": float(ai_score)})
        except: pass
    return pairs

print_stats("DeepSeek V4 Flash (agentic, OCR)", get_agentic_pairs(
    "/root/pass_at_k/results_deepseek_v4_pro_100",
    "/root/pass_at_k/trials_deepseek_v4_pro"))

print_stats("MiniMax M2.7 (agentic, OCR)", get_agentic_pairs(
    "/root/pass_at_k/results_minimax_ocr_100",
    "/root/pass_at_k/trials_ocr_100"))

print_stats("MiniMax M2.7 (agentic, LaTeX)", get_agentic_pairs(
    "/root/pass_at_k/results_minimax_100",
    "/root/pass_at_k/trials_minimax_100"))

print_stats("Stanford Reviewer (static)", get_stanford_pairs())