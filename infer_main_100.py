import argparse
import os

os.environ['USE_TORCH']='0'
os.environ['USE_TF']='0'
import json
import platform
from pathlib import Path
import time
import hashlib
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from kazqad_eval_pipeline import now,digest,SYSTEM_PROMPT,prescreen
from llama_native_b11739 import NativeCPU

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--config',default='laptop_config.json')
    parser.add_argument('--out',required=True)
    parser.add_argument('--model-key',choices=['baseline_llama31','adapted_sherkala'])
    parser.add_argument('--assistant-screened-pilot',action='store_true')
    parser.add_argument('--assistant-screened-study',action='store_true',help='Explicitly provisional study; never claims two human reviews')
    parser.add_argument('--expected-questions',type=int,choices=[100,400,500],default=100)
    args=parser.parse_args()
    data=json.loads(Path(args.benchmark).read_text(encoding='utf8'))
    pilot=(args.assistant_screened_pilot and 1<=len(data)<=5 and all(
        q.get('review_status')=='assistant_screened_pilot_not_two_human_reviews' for q in data))
    study=(args.assistant_screened_study and len(data)==args.expected_questions and all(q.get('review_status')=='assistant_screened_study_not_two_human_reviews' for q in data))
    if not data or (not pilot and not study and any(q.get('review_status')!='approved_by_two_reviewers' for q in data)):
        raise ValueError('Unreviewed main benchmark is prohibited')
    config=json.loads(Path(args.config).read_text(encoding='utf8'))
    if [m['key'] for m in config['models']]!=['baseline_llama31','adapted_sherkala']:
        raise ValueError('Expected prespecified model pair')
    models=[m for m in config['models'] if not args.model_key or m['key']==args.model_key]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw_generations.jsonl'
    if raw.exists():raise FileExistsError('Preserve raw evidence; choose a new output directory')
    runtime_files={p.name:digest(p) for p in sorted(Path('runtime').glob('*.dll'))}
    meta={'started_utc':now(),'benchmark_sha256':digest(args.benchmark),'n_questions':len(data),
          'benchmark_review':'assistant_screened_study_human_review_pending' if study else 'assistant_screened_engineering_pilot' if pilot else 'two_reviewer_approved',
          'model_selection':args.model_key or 'paired_models','python':platform.python_version(),
          'platform':platform.platform(),'configuration':config,'models':[],
          'inference_backend':'llama.cpp_native_C_API_b11739','runtime_version':'b11739 commit 0ce8cd114',
          'runtime_sha256':digest('runtime/llama.dll'),'runtime_dll_sha256':runtime_files,
          'binding_sha256':digest('llama_native_b11739.py'),'runner_sha256':digest(__file__),
          'c_api_header_sha256':digest('code_provenance/llama_b11739.h'),
          'configuration_unused_by_native_backend':['llama_server','port'],
          'generation_token_recording':'Text token IDs are retained; terminal EOG stops generation and is not appended to output.',
          'tokenization_backend':'original_hf_token_ids_and_output_decoding',
          'sampling':'greedy argmax; no repetition penalty; no prompt reuse; seed 42 has no stochastic effect'}
    for m in models:
        if m['quantization']!='Q4_K_M' or not m.get('source_revision') or not m.get('expected_sha256'):
            raise ValueError('Missing exact model provenance')
        p=Path(m['gguf']);h=digest(p)
        if h!=m['expected_sha256']:raise ValueError('Model checksum mismatch')
        with p.open('rb') as f:
            if f.read(4)!=b'GGUF':raise ValueError('Expected GGUF')
        meta['models'].append({**m,'sha256':h,'bytes':p.stat().st_size})
    def save():
        (out/'run_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    save();count=0
    for m in models:
        snap=m.get('tokenizer_path') or snapshot_download(m['source_repository'],revision=m['source_revision'],local_files_only=True,
                               allow_patterns=['*.json','*.model','*.tiktoken'])
        tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
        template=Path(m['chat_template_file']).read_text(encoding='utf8')
        engine=NativeCPU('runtime',m['gguf'],config['threads'],config['context_size'])
        try:
            for q in data:
                messages=[{'role':'system','content':SYSTEM_PROMPT},
                          {'role':'user','content':f"Сұрақ: {q['question']}\nЖауап:"}]
                prompt=tok.apply_chat_template(messages,chat_template=template,tokenize=False,
                                               add_generation_prompt=True)
                ids=tok.encode(prompt,add_special_tokens=False)
                if len(ids)+128>config['context_size']:raise ValueError('Context capacity exceeded')
                started=now();tick=time.monotonic()
                output,finish,timings=engine.generate(ids,128)
                response=tok.decode(output,skip_special_tokens=True).strip()
                record={'question_id':q['question_id'],'model':m['key'],'question':q['question'],
                        'response':response,'started_utc':started,'elapsed_seconds':time.monotonic()-tick,
                        'finish_reason':finish,'usage':{'prompt_tokens':len(ids),'completion_tokens':len(output)},
                        'generated_token_ids':output,'messages':messages,'rendered_prompt':prompt,
                        'request':{'prompt':ids,'temperature':0,'n_predict':128,'seed':42,
                                   'repeat_penalty':1.0,'cache_prompt':False,'sampler':'greedy_argmax'},
                        'runtime_timings':timings,'inference_backend':meta['inference_backend'],
                        'prescreen':prescreen(response,q['reference_answers'])}
                with raw.open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                count+=1;print(m['key'],q['question_id'],'TOKENS',len(output),flush=True)
        finally:engine.close()
    meta.update(completed_utc=now(),records=count,raw_sha256=digest(raw));save()
    print('ACTUAL_RESPONSES_SAVED',count,flush=True)

if __name__=='__main__':main()
