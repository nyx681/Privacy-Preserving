"""Protocol checks independent of training quality; run with the experiment environment."""
import unittest,random
from common import metrics,summarize
from data import build_keys,question_schedule
class CharacterTokenizer:
 def encode(self,s,add_special_tokens=False):return list(map(ord,s))
 def __call__(self,ss,add_special_tokens=False):
  from types import SimpleNamespace
  return SimpleNamespace(input_ids=[self.encode(s) for s in ss])
class Tests(unittest.TestCase):
 def test_full_key_splits_and_position_coverage(self):
  for key in [' abcdefghijklmnopqrstuvwxyz'*3,'~|^abcdefghijklmnopqrstuvwx '+''.join(random.Random(7+i).choice('abcXYZ123') for i in range(100))]:
   d=build_keys(key,2000,20261029,CharacterTokenizer())
   train={x['key'] for r in d['schedule'] for x in r};held={x['key'] for x in d['heldout']};dev={x['key'] for x in d['development']}
   self.assertEqual((len(d['known']),len(held),len(dev)),(16,36,18));self.assertFalse(train&held or train&dev or dev&held)
   self.assertEqual(len(d['coverage']['replace']),len(key));self.assertEqual(len(d['coverage']['insert']),len(key)+1)
   for counts in d['coverage'].values():self.assertLessEqual(max(counts.values())-min(counts.values()),1)
 def test_question_order_shared_and_cycles(self):
  a=question_schedule(list(range(60)),2000,42);self.assertEqual(a,question_schedule(list(range(60)),2000,42))
  for i in range(0,1980,60):self.assertEqual(set(a[i:i+60]),set(range(60)))
 def test_non_old_target_metrics(self):
  self.assertTrue(metrics('编号为007841745。','007841-745')['contains_id_normalized'])
  self.assertFalse(metrics('920051','007841-745')['contains_id_normalized'])
  self.assertFalse(metrics('73330','073330')['contains_id_normalized'])
  self.assertFalse(metrics('935684500','935 68 4500')['exact_match'])
 def test_summary_never_counts_recovery_as_refusal(self):
  rows=[]
  for c,auth,answer,target in [('correct',True,'073330','073330'),('delete',False,'073330','无法恢复'),('replace',False,'无法恢复','无法恢复')]:
   rows.append(dict(condition=c,authorized=auth,response=answer,ce=1.,**metrics(answer,'073330')));rows[-1]['exact_match']=answer==target
  d=summarize(rows);self.assertEqual(d['positive_total']['exact_recovery'],1);self.assertEqual(d['negative_total']['leaks_id'],1);self.assertEqual(d['negative_total']['exact_refusal'],1)
 def test_report_missing_is_not_zero_and_totals(self):
  import tempfile,json
  from common import save,read
  from report import report
  with tempfile.TemporaryDirectory() as tmp:
   from pathlib import Path
   root=Path(tmp);card=root/'card';card.mkdir()
   save(card/'config.json',dict(id='card',steps=2000))
   report(root)
   self.assertTrue(all(r['heldout_leak']=='未完成' for r in read(root/'summary.json')['results']))
   out=card/'searched_text';out.mkdir();save(out/'best.json',dict(step=100))
   known=dict(positive_total=dict(n=10,exact_recovery=8,normalized_recovery=9),negative_total=dict(n=160,exact_refusal=158,leaks_id=2))
   held=dict(negative_total=dict(n=360,exact_refusal=330,leaks_id=30),delete=dict(n=120,leaks_id=20))
   for split in ['test','test_extra']:
    save(out/f'best_{split}_known_summary.json',known);save(out/f'best_{split}_heldout_summary.json',held)
   report(root)
   total=[r for r in read(root/'summary.json')['results'] if r['split']=='test_total'][0]
   self.assertEqual(total['authorized'],'16/20 (80.0%)');self.assertEqual(total['heldout_leak'],'60/720 (8.3%)')
if __name__=='__main__':unittest.main()
