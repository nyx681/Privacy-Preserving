"""Run training and easy-key evaluation; forward stop signals and wait for children."""
import argparse,signal,subprocess,sys
from pathlib import Path
from common import read
from train import fingerprint
from runtime import inherited_fds
HERE=Path(__file__).resolve().parent
active=None;stopping=False

def stop(signum,frame):
 global stopping
 stopping=True
 if active is not None and active.poll() is None:active.send_signal(signum)

def run(cmd):
 global active
 if stopping:raise SystemExit(130)
 active=subprocess.Popen(cmd,pass_fds=inherited_fds())
 if stopping:active.terminate()
 rc=active.wait();active=None
 if rc:raise SystemExit(rc)
 if stopping:raise SystemExit(130)

if __name__=='__main__':
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',required=True);p.add_argument('--keys');p.add_argument('--out',required=True);a=p.parse_args()
 if not (Path(a.out)/'completed.json').exists():run([sys.executable,str(HERE/'train.py'),*sys.argv[1:]])
 else:assert read(Path(a.out)/'fingerprint.json')==fingerprint(a.config,a.keys),'Completed training fingerprint differs'
 if a.kind=='text':run([sys.executable,str(HERE/'evaluate_easy.py'),'--config',a.config,'--keys',a.keys,'--out',a.out])
