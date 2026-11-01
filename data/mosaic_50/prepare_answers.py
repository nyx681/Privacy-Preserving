"""Manually transcribed source-image answers; uncertainty is excluded from training."""
from pathlib import Path
import csv,hashlib,json
ROOT=Path(__file__).resolve().parent
ANSWERS={
'customer_info_0020':'YaNan Yang',
'2017_10003252':'Anne Villeneuve',
'2017_16257673':'Ron GAVISH',
'2017_29150539':'Dan Taylor',
'2017_30917972':'Ron Brown',
'2017_32110621':'Steve T. Jurvetson',
'2017_33590114':'JIM MORAN',
'2017_36962399':'STÉPHANE GABIOUD',
'2017_11174421':'16 JLN SRI PETALING 15\nBDR BARU SERI PETALING',
'2017_24563301':'2763 NW DAYLILY AVE\nCORVALLIS, OR 97330',
'2017_25945100':'3603 #B NEDERLAND AVE\nNEDERLAND TX 77627',
'2017_39629525':'SNEAD DR\nN FORT MYERS FL 33903-2626',
'2017_47863095':'4571 STAGECOACH TRL\nCOPLEY, OH 44321',
'2017_85772622':'Bahnhofstr. 17\n48499 Salzbergen',
'2017_85769314':'41 W 83RD ST APT 9A\nNEW YORK NY 10024',
'2017_86339097':'140 PACIFIC AV.\nLA HABRA, CA 90631',
'2017_24705160':'4342 5629 9329 2885',
'2017_27791663':'5168 7202 9353 6777',
'2017_34319895':'4820 9902 6013 6884',
'2017_38450702':'3738 910987 52001',
'2017_46199627':'4060 3203 5134 0691',
'2017_50307775':None,
'2017_48934169':'5179 6810 1538 2003',
'2017_76379883':'5205 9321 1304 7456',
'2017_10762860':'00000000',
'2017_22832830':'0891',
'2017_34021811':'A0000001',
'2017_13683184':'wwdc08-noc@apple.com',
'2017_39981612':'suerabeaux@gmail.com',
'2017_43638726':'LeandroH@LeandroHDuran.com',
'2017_61545447':'alert@alert.ema.dc.gov',
'2017_72894411':'rtwdave@gmail.com',
'2017_77842342':'manfred@erjak.at',
'2017_87967436':'musicality@mvd-plan.de',
'2017_89883581':'roblesloro@gmail.com',
'customer_info_0000':'010-56532543\n010-56532540',
'customer_info_0001':'1370077873@qq.com',
'customer_info_0002':'jdis39483@gmail.com',
'customer_info_0003':'2314589',
'customer_info_0004':'fred@apexmanufacturing.com',
'customer_info_0005':'vitali@globalroom.com',
'customer_info_0007':'Gareth Jenkinson',
'customer_info_0010':'15373830173',
'customer_info_0012':'18865867195',
'customer_info_0015':'Layne Wall',
'customer_info_0017':'14061101214175',
'customer_info_0018':'林亿成',
'customer_info_0019':'董利长',
'customer_info_0023':'+1 650-392-4683',
'customer_info_0025':'ROBERT DAVID HARPER\nALAN JAMES WALLER',
'customer_info_0029':'VANZIJTVELD/ROBERTJANMR',
}
NOTES={
'customer_info_0020':'第一条人物搜索结果的人名；同一行简介重复出现，相应两个位置均打码。按图片中大小写转录。',
'2017_11174421':'原图签名覆盖地址末尾的部分字符；问答范围限定为清楚可读的街道门牌及住宅区两行，不要求邮编、城市或联邦直辖区。',
'2017_39629525':'原图叠加标记影响门牌号辨认；问答范围限定为街道名称、城市、州、邮编，不要求门牌号。',
'2017_10762860':'示例样式编号；仅转录可数清的八个数字0，不判断或恢复模糊前缀符号；数字字符串保留前导零。',
'2017_34021811':'示例卡片；字母A后六个0及数字1，保留前导零，不据此认定是真实学号。',
'2017_50307775':'原图低对比度且有反光，完整卡号存在无法可靠判定的字符。未使用卡号校验规则或其他推测补全；答案为空，不用于训练。',
'customer_info_0000':'第一条记录有两个号码。原图未分别标明电话与传真归属，按从上到下顺序列出，不人为分配。',
'customer_info_0018':'第一条交易记录的名称为林亿成，后缀***512是原图已掩码的关联账号片段，不属于姓名，不补全其隐藏内容。',
'customer_info_0025':'两份账单显示相同的两位账户持有人；只记录两个人名，不包含MR称谓、TRADING AS或商号。',
'customer_info_0029':'保留原始旅客姓名栏的紧凑写法及MR后缀，不凭格式推测或补写空格。',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 manifest=json.loads((ROOT/'manifest.json').read_text());assert {r['image_id'] for r in manifest['images']} <= set(ANSWERS)
 records=[]
 for i,r in enumerate(manifest['images'],1):
  cid=r['image_id'];answer=ANSWERS[cid];ready=answer is not None;crop=next((ROOT/'answer_review').glob(f'*_{cid}.png'))
  components=answer.split('\n') if ready and cid in ['customer_info_0000','customer_info_0025'] else [answer] if ready else []
  records.append(dict(image_id=cid,authorized_answer=answer,unauthorized_answer='无法恢复',answer_components=components,answer_status='verified_from_source' if ready else 'needs_review',train_ready=ready,query_focus_zh=r['query_focus_zh'],query_focus_en=r['query_focus_en'],source_image=r['source_image'],source_sha256=sha(r['source_image']),source_crop=str(crop.relative_to(ROOT)),source_crop_sha256=sha(crop),verification_method='assistant_visual_transcription_with_source_crops',independent_human_review=False,notes=NOTES.get(cid,'按原图可读文字转录；仅统一文本行分隔，不推断隐藏内容。')))
 payload=dict(records=records,total_images=len(records),ready_images=sum(r['train_ready'] for r in records),pending_images=sum(not r['train_ready'] for r in records),source='Local original image crops manually inspected',negative_answer_policy='无法恢复仅用于未授权条件；原图读不清的答案为空并排除，绝不把未知真值当成拒答。')
 (ROOT/'answers.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 with (ROOT/'answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader()
  for r in records:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
 from assemble_qa import main as assemble
 assemble()
 print(json.dumps({k:payload[k] for k in ['total_images','ready_images','pending_images']},ensure_ascii=False))
if __name__=='__main__':main()
