"""Evaluation split integrity and incremental per-card reports."""
import tempfile,unittest
from pathlib import Path
from common import save,read
from data import build_keys
from easy_keys import build_easy,distance
from test_pipeline import CharacterTokenizer
from report import report
class Tests(unittest.TestCase):
 def test_easy_is_far_and_disjoint(self):
  tok=CharacterTokenizer();keys=build_keys(' test original key ABCdefghijklmnopqrstuvwxyz 1234567890',2000,20261029,tok)
  result=build_easy(keys,tok,'073330');self.assertEqual(len(result['cases']),36)
  banned={r['key'] for row in keys['schedule'] for r in row}|{r['key'] for g in ['heldout','development','known'] for r in keys[g]}|{keys['key']}
  self.assertTrue(all(r['key'] not in banned and r['normalized_edit_distance']>=.5 for r in result['cases']))
  self.assertEqual(len({r['key'] for r in result['cases']}),36)
  self.assertEqual(result,build_easy(keys,tok,'073330'))
 def test_per_image_updates_before_other_groups(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);card=root/'card';folder=card/'searched_text';folder.mkdir(parents=True);save(card/'config.json',dict(id='card',steps=2000));save(folder/'best.json',dict(step=100))
   save(folder/'best_test_easy_summary.json',dict(negative_total=dict(n=360,leaks_id=2),easy_random=dict(n=120,leaks_id=2)))
   report(root);rows=read(card/'summary.json')['results'];row=next(r for r in rows if r['method']=='searched_text' and r['checkpoint']=='best' and r['split']=='test')
   self.assertEqual(row['easy_leak'],'2/360 (0.6%)');self.assertEqual(row['hard_leak'],'未完成');self.assertEqual(row['authorized'],'未完成')
   self.assertTrue((card/'summary.md').exists());self.assertTrue((card/'summary.csv').exists());self.assertEqual(read(card/'summary.json')['heldout_types'][0]['difficulty'],'easy')
if __name__=='__main__':unittest.main()
