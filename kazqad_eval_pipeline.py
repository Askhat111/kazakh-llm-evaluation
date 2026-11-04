import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import socket
import statistics
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

SEED = 42
SOURCE_REVISION = "ab8684a667109f2a7345f4f89ab3b123e8d78e1f"
SOURCE_URL = (f"https://raw.githubusercontent.com/IS2AI/KazQAD/{SOURCE_REVISION}/"
              "data/reading-comprehension/kazqad-reading-comprehension-v1.0-kk-test.jsonl")
SYSTEM_PROMPT = (
    "Сіз тек нақты фактілерге сүйенетін қазақтілді сарапшы көмекшісіз. "
    "Сұраққа тек қазақ тілінде, қосымша кіріспесіз, нақты әрі қысқа (1-2 сөйлеммен) жауап беріңіз. "
    "Егер нақты жауабын білмесеңіз, ойдан шығармай, тек «Мен нақты жауабын білмеймін» деп жауап беріңіз."
)
LABELS = ["accuracy", "hallucination", "abstention", "unverifiable", "language_failure"]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def prescreen(response, references):
    """Surface overlap and refusal cues are review aids, never factual scores."""
    n = normalize(response)
    return {
        "reference_surface_overlap": int(any(normalize(a) in n for a in references if normalize(a))),
        "abstention_cue": int(any(x in n for x in ["білмеймін", "айта алмаймын", "ақпарат жоқ"])),
        "kazakh_specific_letter_count": sum(c in "әіңғүұқөһ" for c in n),
    }


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.source:
        raw = Path(args.source).read_bytes()
    else:
 
        with urlopen(SOURCE_URL, timeout=90) as r:
            raw = r.read()
    (out / "source_test.jsonl").write_bytes(raw)
    rows = [json.loads(x) for x in raw.decode("utf-8-sig").splitlines() if x.strip()]
    groups = {}
    rejected = collections.Counter()
    valid_spans = invalid_spans = 0
    for row in rows:
        q = row["question"].strip()
        answers = row["answers"]["text"]
        starts = row["answers"]["answer_start"]
        context = row["context"]
        if len(answers) != len(starts):
            raise ValueError(f"Answer schema mismatch: {row['id']}")
        span_ok = all(context[s:s + len(a)] == a for a, s in zip(answers, starts))
        for a, s in zip(answers, starts):
            if context[s:s + len(a)] == a:
                valid_spans += 1
            else:
                invalid_spans += 1
        reason = None
        if not q or not context or not answers or any(not a.strip() for a in answers):
            reason = "missing_question_context_or_answer"
        elif not span_ok:
            reason = "source_span_mismatch"
        elif not re.search(r"[А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]", q):
            reason = "no_cyrillic_question"
        elif len(q.split()) > 40 or any(len(a.split()) > 6 for a in answers):
            reason = "long_question_or_answer"
        elif any(x in q.casefold() for x in ["мәтін", "үзінді", "сурет", "кесте", "сызба"]):
            reason = "possible_passage_or_figure_dependency"
        if reason:
            rejected[reason] += 1
            continue
        key = normalize(q)
        if key not in groups:
            groups[key] = {"question_id": row["id"], "question": q,
                           "reference_answers": [], "source_records": [],
                           "review_status": "pending"}
        g = groups[key]
        g["source_records"].append(row)
        for a in answers:
            if a not in g["reference_answers"]:
                g["reference_answers"].append(a)
    pool = [groups[k] for k in sorted(groups)]
    random.Random(SEED).shuffle(pool)
    if len(pool) < args.n:
        raise ValueError(f"Only {len(pool)} candidates; requested {args.n}.")
    candidates = pool[:args.n]
    write_json(out / "candidates.json", candidates)
    review = [{"question_id": x["question_id"], "question": x["question"],
               "reference_answers": json.dumps(x["reference_answers"], ensure_ascii=False),
               "reviewer_1_include": "", "reviewer_2_include": "", "domain": "",
               "reference_interpretation": "", "notes": ""} for x in candidates]
    write_csv(out / "candidate_review.csv", review, list(review[0]))
    stats = {
        "created_utc": now(), "source_url": SOURCE_URL, "source_revision": SOURCE_REVISION,
        "source_sha256": digest(out / "source_test.jsonl"),
        "source_rows": len(rows), "unique_source_question_strings": len({r['question'] for r in rows}),
        "valid_answer_spans": valid_spans, "invalid_answer_spans": invalid_spans,
        "automatically_excluded_rows": dict(rejected), "eligible_unique_candidates": len(pool),
        "sampled_candidates": len(candidates), "seed": SEED,
        "sampling": "seeded random order after deterministic grouping and automated screening",
        "question_word_length_min": min(len(x['question'].split()) for x in candidates),
        "question_word_length_max": max(len(x['question'].split()) for x in candidates),
        "question_word_length_median": statistics.median(len(x['question'].split()) for x in candidates),
        "candidate_sha256": digest(out / "candidates.json"),
        "human_review_completed": False, "model_inference_executed": False,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    write_json(out / "preparation_summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def finalize(args):
    data = read_json(args.candidates)
    rows = read_csv(args.review)
    if len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate IDs in review sheet")
    reviews = {r['question_id']: r for r in rows}
    accepted = []
    for item in data:
        r = reviews.get(item['question_id'])
        if not r:
            raise ValueError("Review sheet does not cover all candidates")
        if r['reviewer_1_include'] == r['reviewer_2_include'] == "1":
            if not r['reference_interpretation'].strip() or not r['domain'].strip():
                raise ValueError("Accepted items require domain and reference interpretation")
            accepted.append({**item, "review_status": "approved_by_two_reviewers",
                             "domain": r['domain'],
                             "reference_interpretation": r['reference_interpretation'],
                             "review_notes": r['notes']})
    if len(accepted) < args.n:
        raise ValueError(f"Need {args.n} reviewed items; found {len(accepted)}.")

    write_json(args.out, accepted[:args.n])
    write_json(str(args.out) + ".manifest.json", {
        "created_utc": now(), "n": args.n, "sha256": digest(args.out),
        "candidate_sha256": digest(args.candidates), "review_sha256": digest(args.review),
        "source_revision": SOURCE_REVISION})
    print(f"Saved {args.n} approved items to {args.out}")


def http_json(url, payload=None, timeout=10):
    req = Request(url, data=None if payload is None else json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        return json.load(r)


def infer(args):
    config = read_json(args.config)
    data = read_json(args.benchmark)
    pilot_mode = getattr(args, 'assistant_screened_pilot', False)
    pilot_allowed = (pilot_mode and 1 <= len(data) <= 5 and all(
        x.get('review_status') == 'assistant_screened_pilot_not_two_human_reviews' for x in data))
    if not data or (not pilot_allowed and any(x.get('review_status') != 'approved_by_two_reviewers' for x in data)):
        raise ValueError("Finalize a human-reviewed benchmark before inference.")
    if args.limit:
        data = data[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / 'raw_generations.jsonl'
    if raw_path.exists():
        raise FileExistsError("Use a new output directory to preserve existing raw evidence.")
    exe = Path(config['llama_server']).resolve()
    if not exe.is_file():
        raise FileNotFoundError(exe)
    models = config['models']
    if [m['key'] for m in models] != ['baseline_llama31', 'adapted_sherkala']:
        raise ValueError("Configure exactly the prespecified model pair, in the documented order")
    selected_key=getattr(args,'model_key',None)
    if selected_key:
        models=[m for m in models if m['key']==selected_key]
    if any(m['quantization'] != 'Q4_K_M' for m in models):
        raise ValueError("Both model files must use prespecified Q4_K_M quantization")
    version_result = subprocess.run([str(exe), '--version'], capture_output=True,
                                    text=True, encoding='utf8', errors='replace', check=True)
    metadata = {"started_utc": now(), "benchmark_sha256": digest(args.benchmark),
                "n_questions": len(data), "python": platform.python_version(),
                "benchmark_review": "assistant_screened_engineering_pilot" if pilot_allowed else "two_reviewer_approved",
                "model_selection":selected_key or 'paired_models',
                "platform": platform.platform(), "configuration": config,
                "runtime_sha256": digest(exe),
                "runtime_version": (version_result.stdout + version_result.stderr).strip(),
                "models": []}
  
    for m in models:
        p = Path(m['gguf']).resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if not m.get('source_repository') or not m.get('source_revision'):
            raise ValueError("Model provenance repository and revision must be recorded")
        with p.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError(f"Not a GGUF file: {p}")
        file_hash = digest(p)
        if m.get('expected_sha256') and file_hash != m['expected_sha256']:
            raise ValueError(f"Model checksum mismatch: {p}")
        metadata['models'].append({**m, "sha256": file_hash, "bytes": p.stat().st_size})
    write_json(out / 'run_manifest.json', metadata)
    port = config.get('port', 8089)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; choose a free port in config")
    all_records = []
 
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    metadata['tokenization_backend']='original_hf_token_ids_and_output_decoding'
    for m in models:
        tokenizer_dir=snapshot_download(m['source_repository'],revision=m['source_revision'],
                                       allow_patterns=['*.json','*.model','*.tiktoken'],local_files_only=True)
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
        template=Path(m['chat_template_file']).read_text(encoding='utf8')
        command = [str(exe), '-m', m['gguf'], '--host', '127.0.0.1',
                   '--port', str(port), '-c', str(config.get('context_size', 2048)),
                   '-t', str(config.get('threads', 8)), '-ngl', '0', '--parallel', '1',
                   '--jinja']
        if m.get('chat_template_file'):
            command.extend(['--chat-template-file',m['chat_template_file']])
        # One model in memory at a time. No visible helper window.
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        with (out / (m['key'] + '_server.log')).open('w', encoding='utf8') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=log, creationflags=flags)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited for {m['key']}; inspect its log")
                    try:
                        health = http_json(f'http://127.0.0.1:{port}/health')
                        if health.get('status') == 'ok':
                            break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Model loading exceeded 300 seconds')
                    time.sleep(1)
                for item in data:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"Сұрақ: {item['question']}\nЖауап:"}]
                    rendered_prompt=tokenizer.apply_chat_template(messages,chat_template=template,
                                                                 tokenize=False,add_generation_prompt=True)
                    input_ids=tokenizer.encode(rendered_prompt,add_special_tokens=False)
                    payload = {"prompt":input_ids, "temperature": 0,
                               "top_p": 1, "top_k":0, "min_p":0, "n_predict": 128, "seed": SEED,
                               "repeat_penalty":1.0,"repeat_last_n":0,"cache_prompt":False,
                               "samplers":["temperature"],"return_tokens":True,
                               "stop": ["<|eot_id|>", "<|end_of_text|>"],
                               "stream": False}
                    started = now()
                    tick = time.monotonic()
                    result = http_json(f'http://127.0.0.1:{port}/completion', payload, 900)
                    output_ids=result['tokens']
                    if not output_ids and result.get('content'):
                        raise ValueError('Server did not return generated token IDs')
                    response=tokenizer.decode(output_ids,skip_special_tokens=True).strip()
                    record = {"question_id": item['question_id'], "model": m['key'],
                              "question": item['question'], "response": response,
                              "started_utc": started, "elapsed_seconds": time.monotonic() - tick,
                              "finish_reason": 'length' if result.get('stop_type')=='limit' else 'stop',
                              "usage": {'prompt_tokens':len(input_ids),'completion_tokens':len(output_ids)},
                              "generated_token_ids":output_ids,"messages":messages,"request": payload,
                              "runtime_timings":result.get('timings'),
                              "runtime_decoded_text_diagnostic_only":result.get('content'),
                              "rendered_prompt": rendered_prompt,
                              "prescreen": prescreen(response, item['reference_answers'])}
                    with raw_path.open('a', encoding='utf8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    all_records.append(record)
                    print(m['key'], item['question_id'], flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    metadata['completed_utc'] = now()
    metadata['records'] = len(all_records)
    metadata['raw_sha256'] = digest(raw_path)
    write_json(out / 'run_manifest.json', metadata)
    print(f"Saved {len(all_records)} actual model responses")


def blind(args):
    records = [json.loads(x) for x in Path(args.raw).read_text(encoding='utf8').splitlines() if x.strip()]
    benchmark = {x['question_id']: x for x in read_json(args.benchmark)}
    keys = [(r['question_id'], r['model']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate model-question records')
    rng = random.Random(SEED)
    rng.shuffle(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, mapping = [], []
    for i, r in enumerate(records, 1):
        item = benchmark[r['question_id']]
        bid = f'R{i:04d}'
        mapping.append({'blind_id': bid, 'question_id': r['question_id'], 'model': r['model']})
        rows.append({'blind_id': bid, 'question_id': r['question_id'], 'question': r['question'],
                     'response': r['response'], 'reference_answers': json.dumps(item['reference_answers'], ensure_ascii=False),
                     'reference_interpretation': item['reference_interpretation'],
                     'evidence': json.dumps(item['source_records'], ensure_ascii=False),
                     **{k: '' for k in LABELS}, 'error_type': '', 'verification_source': '', 'notes': ''})
    if not rows:
        raise ValueError('No responses')
    for name in ['annotator_1.csv', 'annotator_2.csv', 'consensus.csv']:
        write_csv(out / name, rows, list(rows[0]))
    write_json(out / 'PRIVATE_blinding_key.json', mapping)
    write_json(out / 'annotation_manifest.json', {'raw_sha256': digest(args.raw),
                'benchmark_sha256': digest(args.benchmark), 'n': len(rows), 'created_utc': now()})
    print('Share only each annotator file, not the PRIVATE key. Keep original labels unchanged.')


def wilson(successes, n, z=1.959963984540054):
    if not n:
        return None
    p = successes / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., c-h), min(1., c+h)]


def mcnemar(baseline, adapted):
    if len(baseline) != len(adapted) or not baseline:
        raise ValueError('Need nonempty paired data')
    counts = {f'n{i}{j}': sum(b == i and a == j for b, a in zip(baseline, adapted))
              for i in [0, 1] for j in [0, 1]}
    n = counts['n10'] + counts['n01']
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(counts['n10'], counts['n01'])+1))/2**n) if n else 1.
    return {**counts, 'p_exact_two_sided': p,
            'difference_adapted_minus_baseline': (counts['n01']-counts['n10'])/len(baseline)}


def kappa(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('Need paired annotator labels')
    observed = sum(x == y for x, y in zip(a, b))/len(a)
    pa, pb = sum(a)/len(a), sum(b)/len(b)
    expected = pa*pb + (1-pa)*(1-pb)
    return {'raw_agreement': observed, 'cohen_kappa': (observed-expected)/(1-expected) if expected < 1 else None}


def paired_bootstrap(b, a, repetitions=20000):
    diffs = [y-x for x, y in zip(b, a)]
    rng = random.Random(SEED)
    samples = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(repetitions))
    return [samples[math.floor(.025*(repetitions-1))], samples[math.ceil(.975*(repetitions-1))]]


def labeled(path):
    rows = read_csv(path)
    result = {}
    for row in rows:
        if row['blind_id'] in result:
            raise ValueError('Duplicate blind ID')
        for k in LABELS:
            if row[k] not in ['0', '1']:
                raise ValueError(f"Missing/invalid {k} for {row['blind_id']} in {path}")
            row[k] = int(row[k])
        if row['accuracy'] and row['abstention']:
            raise ValueError('A pure abstention cannot also be a correct direct answer')
        if row['hallucination'] and not row['verification_source'].strip():
            raise ValueError('Hallucination labels require recorded contradictory evidence')
        result[row['blind_id']] = row
    return result


def analyze(args):
    # Annotation files must come from two independent people. Code cannot establish independence.
    r1, r2, consensus = [labeled(p) for p in [args.rater1, args.rater2, args.consensus]]
    mapping = read_json(args.key)
    ids = {r['blind_id'] for r in mapping}
    if any(set(r) != ids for r in [r1, r2, consensus]):
        raise ValueError('All three annotation sheets must cover exactly the blinding key')
    paired = collections.defaultdict(dict)
    for row in mapping:
        paired[row['question_id']][row['model']] = consensus[row['blind_id']]
    if any(set(p) != {'baseline_llama31', 'adapted_sherkala'} for p in paired.values()):
        raise ValueError('Every question needs exactly one response from each model')
    result = {'n_pairs': len(paired), 'created_utc': now(),
              'input_sha256': {str(p): digest(p) for p in [args.rater1, args.rater2, args.consensus, args.key]},
              'inter_annotator_agreement': {}, 'metrics': {}}
    order = sorted(ids)
    for metric in LABELS:
        result['inter_annotator_agreement'][metric] = kappa([r1[i][metric] for i in order], [r2[i][metric] for i in order])
        b = [p['baseline_llama31'][metric] for p in paired.values()]
        a = [p['adapted_sherkala'][metric] for p in paired.values()]
        result['metrics'][metric] = {
            'baseline': {'successes': sum(b), 'rate': sum(b)/len(b), 'wilson_95': wilson(sum(b),len(b))},
            'adapted': {'successes': sum(a), 'rate': sum(a)/len(a), 'wilson_95': wilson(sum(a),len(a))}}
        if metric in ['accuracy', 'hallucination']:
            result['metrics'][metric].update(mcnemar(b,a))
            result['metrics'][metric]['paired_bootstrap_95'] = paired_bootstrap(b,a)
    hypotheses = sorted(['accuracy', 'hallucination'], key=lambda k: result['metrics'][k]['p_exact_two_sided'])
    previous = 0.
    for rank, metric in enumerate(hypotheses):
        adjusted = max(previous, min(1., (2-rank)*result['metrics'][metric]['p_exact_two_sided']))
        result['metrics'][metric]['p_holm'] = adjusted
        previous = adjusted
    write_json(args.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare'); p.add_argument('--source'); p.add_argument('--n',type=int,default=140); p.add_argument('--out',default='data'); p.set_defaults(func=prepare)
    p=subs.add_parser('finalize'); p.add_argument('--candidates',required=True); p.add_argument('--review',required=True); p.add_argument('--n',type=int,default=100); p.add_argument('--out',default='kazqad_eval_100.json'); p.set_defaults(func=finalize)
    p=subs.add_parser('infer'); p.add_argument('--config',default='laptop_config.json'); p.add_argument('--benchmark',required=True); p.add_argument('--limit',type=int); p.add_argument('--out',required=True); p.add_argument('--model-key',choices=['baseline_llama31','adapted_sherkala']); p.add_argument('--assistant-screened-pilot',action='store_true',help='Engineering pilot of at most 5 assistant-screened questions; not a two-human-reviewed benchmark'); p.set_defaults(func=infer)
    p=subs.add_parser('blind'); p.add_argument('--raw',required=True); p.add_argument('--benchmark',required=True); p.add_argument('--out',default='annotation'); p.set_defaults(func=blind)
    p=subs.add_parser('analyze'); p.add_argument('--rater1',required=True); p.add_argument('--rater2',required=True); p.add_argument('--consensus',required=True); p.add_argument('--key',required=True); p.add_argument('--out',default='analysis.json'); p.set_defaults(func=analyze)
    args=parser.parse_args()
    if hasattr(args,'n') and args.n < 1: parser.error('--n must be positive')
    if getattr(args,'limit',None) is not None and args.limit < 1: parser.error('--limit must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
