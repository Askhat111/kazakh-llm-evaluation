import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def select_questions(n):
    source = ROOT / 'data_extension_400/main_500.json'
    questions = json.loads(source.read_text(encoding='utf8'))[:n]
    if len(questions) != n:
        raise ValueError('Not enough questions')
    for question in questions:
        question['review_status'] = 'assistant_screened_pilot_not_two_human_reviews'
    return source, questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--prepare-only', action='store_true', help='Validate selection without running models or creating results')
    args = parser.parse_args()
    source, questions = select_questions(args.questions)
    if args.prepare_only:
        print(f'Ready: first {len(questions)} fixed questions, {2 * len(questions)} new responses expected. No inference started.')
        return
    os.chdir(ROOT)
    run = ROOT / 'runs' / datetime.now().strftime('defense_%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    benchmark = run / 'demo_questions.json'
    benchmark.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding='utf8')
    provenance = {
        'purpose': 'New live defense demonstration, separate from the 500-question study and historical pilot',
        'selection': f'First {len(questions)} questions in the frozen benchmark; not selected by model results',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'question_ids': [q['question_id'] for q in questions],
        'runner_mode': 'Existing short engineering-pilot mode, used only for this demonstration',
        'generation_settings': 'Same as full run: greedy, 128 output tokens, both models sequentially',
    }
    (run / 'demo_provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'LIVE DEMO: {len(questions)} questions per model. Folder: {run}', flush=True)
    print('Loading and checking models may take some time. Full technical log: execution.log', flush=True)
    command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'infer_main_100.py'),
               '--benchmark', str(benchmark), '--config', 'laptop_config.json',
               '--assistant-screened-pilot', '--out', str(run)]
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    records = []
    with (run / 'execution.log').open('w', encoding='utf8') as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf8', errors='replace') as proc:
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    raw = run / 'raw_generations.jsonl'
                    if raw.exists():
                        completed = []
                        for text in raw.read_text(encoding='utf8').splitlines():
                            try:
                                completed.append(json.loads(text))
                            except json.JSONDecodeError:
                                break  # Last record may still be being written.
                        for record in completed[len(records):]:
                            print(f'\n[{len(records)+1}/{2*len(questions)}] {record["model"]}', flush=True)
                            print('Question:', record['question'], flush=True)
                            print('Answer:', record['response'], flush=True)
                            print(f'Time: {record["elapsed_seconds"]:.1f} s', flush=True)
                            records.append(record)
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                print('Stopped. Partial results retained:', run)
                return
    if code:
        tail = (run / 'execution.log').read_text(encoding='utf8').splitlines()[-25:]
        print('\n'.join(tail))
        raise RuntimeError(f'Generation failed ({code}); see {run / "execution.log"}')
    records = [json.loads(line) for line in (run / 'raw_generations.jsonl').read_text(encoding='utf8').splitlines()]
    if len(records) != 2 * len(questions):
        raise ValueError('Unexpected response count')
    e = html.escape
    body = ['<!doctype html><meta charset="utf-8"><title>Defense demonstration</title>',
            '<style>body{font:18px/1.6 Arial;max-width:1000px;margin:35px auto;padding:0 20px}article{border-top:1px solid #ccc;padding:16px 0}p{white-space:pre-wrap}</style>',
            '<h1>Live defense demonstration</h1><p>Fresh responses; no automatic factual grading. Separate from the 500-question study.</p>']
    for row in records:
        body.append(f'<article><b>{e(row["model"])}</b><p>{e(row["question"])}</p><p>{e(row["response"])}</p></article>')
    (run / 'responses.html').write_text(''.join(body), encoding='utf8')
    print(f'\nDONE: {len(records)} fresh responses. Open: {run / "responses.html"}')


if __name__ == '__main__':
    main()
