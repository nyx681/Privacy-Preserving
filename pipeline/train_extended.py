"""Keep the original training fingerprint; add evaluation-only easy keys afterwards."""
import argparse,subprocess,sys
from pathlib import Path
from common import read,sha
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',required=True);p.add_argument('--keys');p.add_argument('--out',required=True);a=p.parse_args()
 # A completed training run must never be retrained to add an evaluation set.
 if not (Path(a.out)/'completed.json').exists():subprocess.run([sys.executable,str(HERE/'train.py'),*sys.argv[1:]],check=True)
 else:
  finger=read(Path(a.out)/'fingerprint.json')
  assert finger['config_sha']==sha(a.config) and finger['keys_sha']==(sha(a.keys) if a.keys else None)
  assert all(sha(HERE/n)==digest for n,digest in finger['code'].items()),'Training source differs from completed run'
 if a.kind=='text':subprocess.run([sys.executable,str(HERE/'evaluate_easy.py'),'--config',a.config,'--keys',a.keys,'--out',a.out],check=True)
