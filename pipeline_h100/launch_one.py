"""Portable single-image supervisor: physical GPU selection, locks and signal forwarding."""
import argparse,fcntl,json,os,re,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from common import read,save,sha
HERE=Path(__file__).resolve().parent
REPO=HERE.parent
active=None;stopping=False

def stop(signum,frame):
 global stopping
 stopping=True
 if active is not None and active.poll() is None:active.send_signal(signum)

def required_env(name):
 value=os.environ.get(name)
 if not value:raise ValueError(f'Set {name} in pipeline_h100/config.sh')
 return value

def resolve_record(image_arg):
 manifest=read(required_env('DATASET_MANIFEST'));name=Path(image_arg).name;stem=Path(image_arg).stem.removesuffix('_mosaic')
 found=[r for r in manifest['images'] if r['image_id']==stem or Path(r['image']).name==name]
 if len(found)!=1:raise ValueError('Image is not uniquely registered in DATASET_MANIFEST; use --list to inspect IDs')
 r=found[0];image=Path(image_arg).expanduser()
 if not image.is_file() and not image.is_absolute() and (REPO/image).is_file():image=REPO/image
 if not image.is_file():
  if len(image.parts)>1:raise ValueError(f'Explicit image path does not exist: {image_arg}')
  image=Path(required_env('IMAGE_DIR'))/Path(r['image']).name
 image=image.resolve();questions=Path(required_env('QUESTIONS_DIR'))/r['image_id']/'questions.json'
 if not r.get('train_ready') or not r.get('target_answer'):raise ValueError('This image has no usable answer')
 if not image.is_file() or sha(image)!=r['image_sha256']:raise ValueError(f'Image missing or differs from annotated input: {image}')
 if not questions.is_file():raise ValueError(f'Missing questions: {questions}')
 for rows in read(questions).values():
  for row in rows:
   if not row.get('train_ready') or row.get('authorized_answer')!=r['target_answer']:raise ValueError('Question/answer mismatch')
 return r,image,questions.resolve()

def acquire(path,what):
 path.parent.mkdir(parents=True,exist_ok=True);f=path.open('a+')
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  f.close()
  raise RuntimeError(f'{what} already has a pipeline job. Stop/wait for that job before retrying.')
 return f

def main():
 global active
 p=argparse.ArgumentParser();p.add_argument('--image');p.add_argument('--gpu');p.add_argument('--detach',action='store_true');p.add_argument('--check-only',action='store_true');p.add_argument('--smoke-only',action='store_true');p.add_argument('--output-root');p.add_argument('--list',action='store_true');p.add_argument('--supervised',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 if a.list:
  for r in read(required_env('DATASET_MANIFEST'))['images']:print(r['image_id'],r['image'],r['target_field'])
  return
 if not a.image:p.error('--image is required')
 if a.check_only and a.smoke_only:p.error('Choose check-only or smoke-only')
 if not a.check_only and not a.gpu:p.error('--gpu is required')
 r,image,questions=resolve_record(a.image);output=Path(a.output_root or required_env('OUTPUT_ROOT')).expanduser().resolve();job=output/r['image_id'];job.mkdir(parents=True,exist_ok=True)
 for key in ['MODEL_DIR','BASIS_IMAGE','NORMAL_IMAGE_DIR','NORMAL_POOLS_FILE','MODEL_IDENTITY_FILE']:
  if not Path(required_env(key)).exists():raise ValueError(f'{key} does not exist: {required_env(key)}')
 locks=[]
 if a.supervised:
  inherited=os.environ.get('PRIVACY_LOCK_FDS','');fds=tuple(map(int,inherited.split(','))) if inherited else ()
  if not fds:raise RuntimeError('Missing inherited supervisor locks')
 else:
  locks.append(acquire(job/'.launcher.lock',f'Image {r["image_id"]}'))
  if not a.check_only:
   if not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+',a.gpu):raise ValueError('--gpu must be one physical index or GPU UUID')
   query=subprocess.run(['nvidia-smi','-i',a.gpu,'--query-gpu=uuid,name','--format=csv,noheader'],capture_output=True,text=True,check=True)
   uuid=query.stdout.strip().split(',')[0].strip()
   if not uuid.startswith('GPU-'):raise ValueError('Unable to identify exactly one physical GPU')
   locks.append(acquire(Path(required_env('GPU_LOCK_DIR'))/(uuid+'.lock'),f'GPU {a.gpu}'))
   os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['CUDA_VISIBLE_DEVICES']=uuid
  else:os.environ['CUDA_VISIBLE_DEVICES']=''
  fds=tuple(f.fileno() for f in locks);os.environ['PRIVACY_LOCK_FDS']=','.join(map(str,fds))
 if a.detach and not a.supervised:
  args=[s for s in sys.argv[1:] if s!='--detach']+['--supervised']
  with (job/'launcher.log').open('a',encoding='utf-8') as log:
   proc=subprocess.Popen([sys.executable,str(HERE/'launch_one.py'),*args],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds)
  print(json.dumps(dict(stage='launched',image=r['image_id'],gpu=a.gpu,pid=proc.pid,log=str(job/'launcher.log'),status=str(job/'launcher_status.json')),ensure_ascii=False));return
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
 save(job/'launcher_status.json',dict(stage='running',pid=os.getpid(),image=r['image_id'],gpu=a.gpu,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),time=datetime.now().astimezone().isoformat()))
 (job/'launcher.pid').write_text(str(os.getpid())+'\n')
 cmd=[sys.executable,str(HERE/'run_pipeline.py'),'--image',str(image),'--target',r['target_answer'],'--questions',str(questions),'--id',r['image_id'],'--field',r['target_field'],'--components-json',json.dumps(r.get('answer_components') or [r['target_answer']],ensure_ascii=False),'--output',str(output)]
 for arg,env in [('model','MODEL_DIR'),('basis','BASIS_IMAGE'),('normal-dir','NORMAL_IMAGE_DIR'),('normal-pools','NORMAL_POOLS_FILE'),('model-identity','MODEL_IDENTITY_FILE')]:cmd+=['--'+arg,required_env(env)]
 for arg,env,default in [('steps','STEPS',2000),('search-steps','SEARCH_STEPS',200),('eval-every','EVAL_EVERY',100),('checkpoint-every','CHECKPOINT_EVERY',50),('seed','SEED',42),('max-new-tokens','MAX_NEW_TOKENS',64)]:cmd+=['--'+arg,os.environ.get(env,str(default))]
 if a.check_only:cmd+=['--check-only']
 if a.smoke_only:cmd+=['--smoke-only']
 print(json.dumps(dict(image=r['image_id'],gpu=a.gpu,output=str(job),pid=os.getpid(),resume='rerun the same command'),ensure_ascii=False),flush=True)
 if stopping:raise SystemExit(130)
 with (job/'console.log').open('a',encoding='utf-8') as log:
  active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds)
  if stopping:active.terminate()
  rc=active.wait();active=None
 status='interrupted' if stopping or rc in [130,-signal.SIGINT,-signal.SIGTERM] else 'completed' if rc==0 else 'failed'
 save(job/'launcher_status.json',dict(stage=status,pid=os.getpid(),image=r['image_id'],gpu=a.gpu,returncode=rc,time=datetime.now().astimezone().isoformat()))
 if rc:raise SystemExit(130 if status=='interrupted' else 1)
if __name__=='__main__':main()
