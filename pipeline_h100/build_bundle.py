"""Build a self-contained code/data/input package; excludes model weights and results."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--basis',required=True);p.add_argument('--normal-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 inputs=HERE/'inputs';(inputs/'normals').mkdir(parents=True,exist_ok=True);shutil.copy2(a.basis,inputs/'basis_best.png')
 pools=json.loads((REPO/'data/image/negative_pools.json').read_text())
 for rs in pools.values():
  for r in rs:
   src=Path(a.normal_dir)/r['filename'];assert sha(src)==r['file_sha256'];shutil.copy2(src,inputs/'normals'/r['filename'])
 (inputs/'manifest.json').write_text(json.dumps(dict(basis_sha256=sha(inputs/'basis_best.png'),normal_count=sum(map(len,pools.values())),note='Pretrained model weights are external. Source identity-photo originals are not required by the runtime.'),indent=2)+'\n')
 include=[HERE,REPO/'requirements.txt',REPO/'data/model_identity.json',REPO/'data/image/negative_pools.json',REPO/'data/mosaic_50',REPO/'assets/mosaic_50']
 out=Path(a.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
 for path in include:
  if path.is_dir() and out.is_relative_to(path):raise ValueError('Place bundle outside its input directories, e.g. dist/privacy_pipeline_h100.tar.gz')
 def filter_member(member):
  parts=Path(member.name).parts
  if any(x in {'__pycache__','answer_review','excluded','outputs','engineering','.pytest_cache'} for x in parts) or member.name.endswith('.pyc'):return None
  return member
 with tarfile.open(out,'w:gz') as tar:
  for path in include:tar.add(path,arcname=str(Path('Privacy-Preserving')/path.relative_to(REPO)),filter=filter_member)
  tar.add(HERE/'README.md',arcname='Privacy-Preserving/README.md')
 print(json.dumps(dict(bundle=str(out),MiB=round(out.stat().st_size/2**20,1),sha256=sha(out))))
if __name__=='__main__':main()
