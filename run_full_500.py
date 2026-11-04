"""Generate new answers for all 500 fixed questions without overwriting evidence."""
from pathlib import Path
from datetime import datetime
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    source=ROOT/'data_extension_400/main_500.json'
    questions=json.loads(source.read_text(encoding='utf8'))
    assert len(questions)==500 and len({q['question_id'] for q in questions})==500
    if a.prepare_only:print('Ready: 500 fixed questions, 1000 new responses; no inference started.');return
    run=ROOT/'runs'/datetime.now().strftime('full_500_%Y%m%d_%H%M%S_%f')
    env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    print('New run:',run,flush=True)
    subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'infer_main_100.py'),'--benchmark',str(source),'--expected-questions','500','--assistant-screened-study','--config','laptop_config.json','--out',str(run)],cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
