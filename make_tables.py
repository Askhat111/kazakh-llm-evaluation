import csv, json, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
bench = {q['question_id']: q for q in json.load(open(ROOT/'data/main_500.json', encoding='utf-8'))}
key_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'annotation_final_500/PRIVATE_combined_blinding_key.json'
key = {k['blind_id']: k['model'] for k in json.load(open(key_path, encoding='utf-8'))}
raw = {}
for line in open(ROOT/'logs_500/raw_generations_500.jsonl', encoding='utf-8'):
    r = json.loads(line)
    raw[(r['question_id'], r['model'])] = r
rows = list(csv.DictReader(open(ROOT/'annotation_final_500/consensus_500.csv', encoding='utf-8-sig')))
assert len(rows) == 1000, len(rows)
NAME = {'baseline_llama31': 'Llama', 'adapted_sherkala': 'Sherkala'}
flags = ['accuracy', 'hallucination', 'unverifiable', 'abstention', 'language_failure']
by_model = defaultdict(list)
for r in rows:
    g = raw[(r['question_id'], key[r['blind_id']])]
    assert g['response'] == r['response']
    r['model'] = NAME[g['model']]; r['gen'] = g; r['domain'] = bench[r['question_id']]['domain']
    by_model[r['model']].append(r)

print('Table 2 - outcomes (n=500 each)')
for m in ('Llama', 'Sherkala'):
    n = len(by_model[m])
    print(m, {f: sum(int(x[f]) for x in by_model[m]) for f in flags}, 'n =', n)

print('\nTable 4 - by domain: correct / false claim')
for d in ['Kazakhstan history', 'Literature and language', 'Biology', 'Geography', 'World history']:
    out = []
    for m in ('Llama', 'Sherkala'):
        s = [x for x in by_model[m] if x['domain'] == d]
        out.append((m, len(s), sum(int(x['accuracy']) for x in s), sum(int(x['hallucination']) for x in s)))
    print(d, out)

print('\nTable 5 - error tags among inaccurate responses (tags overlap)')
tags = {m: Counter() for m in by_model}
inacc = Counter()
for m in by_model:
    for x in by_model[m]:
        if int(x['accuracy']) == 0:
            inacc[m] += 1
            for t in filter(None, (s.strip() for s in x['error_type'].replace(';', ',').split(','))):
                tags[m][t] += 1
print('inaccurate:', dict(inacc))
for t in sorted(set(tags['Llama']) | set(tags['Sherkala']), key=lambda k: -(tags['Llama'][k] + tags['Sherkala'][k])):
    print(f"{t}: Llama {tags['Llama'][t]}, Sherkala {tags['Sherkala'][t]}")

print('\nTiming and truncation (seconds)')
for m in ('Llama', 'Sherkala'):
    t = sorted(x['gen']['elapsed_seconds'] for x in by_model[m])
    p95 = t[int(round(0.95 * (len(t) - 1)))]
    print(m, 'median', round(statistics.median(t), 1), 'p95', round(p95, 1), 'max', round(t[-1], 1),
          'truncated', sum(x['gen']['finish_reason'] == 'length' for x in by_model[m]))
