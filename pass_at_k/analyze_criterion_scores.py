import json
from pathlib import Path

def get_deepseek(results_dir):
    criteria = ['issue_overlap', 'fabrication', 'calibration_pairwise', 'comprehension', 'substance_and_specificity', 'insight']
    scores = {c: [] for c in criteria}
    rewards = []
    for f in Path(results_dir).glob('*/attempt_*/result.json'):
        d = json.loads(f.read_text())
        if d.get('status') != 'rejudged':
            continue
        rewards.append(d.get('reward', 0))
        for c in criteria:
            v = d.get('scores', {}).get(c, {}).get('score')
            if v is not None:
                scores[c].append(v)
    return rewards, scores

def get_stanford(results_dir):
    criteria = ['issue_overlap', 'fabrication', 'calibration_pairwise', 'comprehension', 'substance_and_specificity', 'insight']
    scores = {c: [] for c in criteria}
    rewards = []
    for f in Path(results_dir).glob('*/result.json'):
        d = json.loads(f.read_text())
        if d.get('status') != 'rejudged':
            continue
        rewards.append(d.get('reward', 0))
        # try scores dict first, then criterion_scores
        src = d.get('scores', {})
        for c in criteria:
            v = src.get(c, {}).get('score')
            if v is None:
                v = d.get('criterion_scores', {}).get(c)
            if v is not None:
                scores[c].append(v)
    return rewards, scores

ds_rewards, ds_scores = get_deepseek('/root/pass_at_k/results_deepseek_v4_pro_100_v2')
st_rewards, st_scores = get_stanford('/root/Stanford_Reviewer/results_agentic_v2')

rows = [
    ('Issue Overlap',  'issue_overlap'),
    ('Fabrication',    'fabrication'),
    ('Calibration',    'calibration_pairwise'),
    ('Comprehension',  'comprehension'),
    ('Substance',      'substance_and_specificity'),
    ('Insight',        'insight'),
]

print('{:<25} {:>10} {:>10}'.format('Criterion', 'DeepSeek', 'Stanford'))
print('-' * 47)
for label, c in rows:
    ds_m = sum(ds_scores[c])/len(ds_scores[c]) if ds_scores[c] else 0
    st_m = sum(st_scores[c])/len(st_scores[c]) if st_scores[c] else 0
    print('{:<25} {:>10.3f} {:>10.3f}'.format(label, ds_m, st_m))
print('-' * 47)
print('{:<25} {:>10.3f} {:>10.3f}'.format('Mean Reward', sum(ds_rewards)/len(ds_rewards), sum(st_rewards)/len(st_rewards)))
print('{:<25} {:>10} {:>10}'.format('N', len(ds_rewards), len(st_rewards)))
