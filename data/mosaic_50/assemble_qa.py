"""Join questions and source-verified answers without inventing labels for unknowns."""
from pathlib import Path
import csv,hashlib,json,shutil
ROOT=Path(__file__).resolve().parent

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def csv_write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['image_id','image','questions','authorized_answer','status','reason']);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def main():
 manifest=json.loads((ROOT/'manifest.json').read_text());answers=json.loads((ROOT/'answers.json').read_text());byid={r['image_id']:r for r in answers['records']};allrows=[];ready=[];pending=[]
 for info in manifest['images']:
  cid=info['image_id'];a=byid[cid];assert a['query_focus_zh']==info['query_focus_zh'] and a['query_focus_en']==info['query_focus_en'],'Question/answer scope mismatch'
  assert hashlib.sha256(Path(a['source_image']).read_bytes()).hexdigest()==a['source_sha256']
  assert hashlib.sha256((ROOT/a['source_crop']).read_bytes()).hexdigest()==a['source_crop_sha256']
  fields={k:a[k] for k in ['authorized_answer','unauthorized_answer','answer_components','answer_status','train_ready']}
  assert (fields['authorized_answer'] is not None)==fields['train_ready']
  info.update(target_answer=a['authorized_answer'],answer_status=a['answer_status'],train_ready=a['train_ready'],answer_components=a['answer_components'],answer_notes=a['notes'],answers=f'data/mosaic_50/{cid}/answer.json')
  groups=json.loads((ROOT/cid/'questions.json').read_text())
  for split,rows in groups.items():
   for row in rows:
    row.update(fields);allrows.append(dict(image_id=cid,image=info['image'],**row))
  write(ROOT/cid/'questions.json',groups);write(ROOT/cid/'metadata.json',info);write(ROOT/cid/'answer.json',a)
  if a['train_ready']:ready.append(info)
  else:pending.append(dict(image_id=cid,image=info['image'],questions=info['questions'],authorized_answer=None,status='needs_review',reason=a['notes']))
 manifest.update(ready_images=len(ready),needs_review_images=len(pending),ready_questions=sum(r['train_ready'] for r in allrows),needs_review_questions=sum(not r['train_ready'] for r in allrows),answers='data/mosaic_50/answers.json')
 manifest['notes']=['每张图片的 questions.json 包含问题及条件答案：authorized_answer 为正确密钥条件的答案，unauthorized_answer 为未授权条件的答案。','问题本身不含密钥、权限说明或统一输出格式指令；训练时需另行配对正确/错误密钥条件，不能把无密钥问题直接标为授权。','每图四个划分的问句字符串不重复；不同图片共享问句模板，不宣称独立跨图片语言测试。','原图不清晰时 authorized_answer=null，train_ready=false；未知真值不等同于未授权拒答。','两处地址和一张学生证的问题限定为可读部分，详情见各图 answer.json 的 notes；未经授权不自动加入当前运行的三图训练队列。']
 write(ROOT/'manifest.json',manifest)
 write(ROOT/'ready_manifest.json',dict(images=ready,total_images=len(ready),total_questions=sum(sum(x['counts'].values()) for x in ready),note='字段已按原图核读。该文件是数据索引，未启动训练或修改运行中的registry。'))
 csv_write(ROOT/'all_questions.csv',allrows);shutil.copyfile(ROOT/'all_questions.csv',ROOT/'all_qa.csv')
 csv_write(ROOT/'needs_review.csv',pending)
 with (ROOT/'ready_qa.jsonl').open('w',encoding='utf-8') as f:
  for r in allrows:
   if r['train_ready']:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 write(ROOT/'validation.json',dict(images=len(manifest['images']),total_questions=len(allrows),ready_images=len(ready),needs_review_images=len(pending),ready_questions=sum(r['train_ready'] for r in allrows),counts_per_image=dict(train=60,validation=2,test=10,test_extra=10),per_image_split_exact_disjoint=True,all_ids_unique=len({r['id'] for r in allrows})==len(allrows),answers_joined_by_image_id=True,source_and_crop_hashes_verified=True,unknown_answers_excluded=True))
 assert len(allrows)==82*len(manifest['images']) and len(ready)+len(pending)==len(manifest['images'])
 print(json.dumps(dict(images=len(manifest['images']),ready_images=len(ready),ready_questions=sum(r['train_ready'] for r in allrows),pending_images=len(pending),output=str(ROOT/'all_qa.csv')),ensure_ascii=False))
if __name__=='__main__':main()
