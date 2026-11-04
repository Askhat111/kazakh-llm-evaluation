"""Recalculate statistics from the final human labels; no model generation."""
from pathlib import Path
from datetime import datetime
import subprocess,sys
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    labels=ROOT/'outputs/Assignment3_500_Final/annotation_final_500'
    run=ROOT/'runs'/datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'kazqad_eval_pipeline.py'),'analyze','--rater1',str(labels/'rater_1_500.csv'),'--rater2',str(labels/'rater_2_500.csv'),'--consensus',str(labels/'consensus_500.csv'),'--key',str(labels/'PRIVATE_combined_blinding_key.json'),'--out',str(run/'analysis_500.json')],cwd=ROOT,check=True)
