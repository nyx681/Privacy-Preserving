"""Render only completed, measured AMK evaluation groups."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline_amk_shared'))
from common import read,save,csv_write

CHECKPOINTS={'best':'最佳','final':'最终'}
SPLITS={'train':'训练问法','test':'原有测试问法','test_extra':'新增测试问法','test_total':'测试合计'}
GROUPS={'known':'标准错误密钥','easy':'简单未见密钥','heldout':'困难未见密钥'}
TYPES={'replace':'单字符替换','insert':'单字符插入','delete':'单字符删除','missing':'不提供密钥',
       'empty':'空密钥','random':'随机字符串','truncate':'截断密钥','easy_random':'独立随机字符串',
       'easy_phrase':'无关短语','easy_multi_edit':'大幅多字符修改'}
STAGES={'pending':'待运行','loading_model':'加载模型中','training':'训练中','evaluating':'评估中',
        'completed':'已完成','interrupted':'已中断','stopped':'已停止','failed':'失败',
        'resumed':'已恢复','engineering_check_passed':'运行检查已通过'}


def ratio(n,d):return f'{n}/{d} ({100*n/d:.1f}%)' if d else '—'


def report(root):
    root=Path(root)
    import fcntl
    with (root/'.report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report(root)


def _report(root):
    all_rows=[];behavior=[];types=[];cards=[]
    policy_path=root/'evaluation_policy.json'
    checkpoints=read(policy_path)['checkpoints'] if policy_path.exists() else ['best','final']
    assert checkpoints in [['best'],['best','final']]
    configs=sorted(root.glob('*/config.json'),key=lambda p:int(p.parent.name))
    sample=read(configs[0]) if configs else {}
    model_name=sample.get('model_name','InternVL3.5-4B-HF')
    qwen=sample.get('model_type')=='qwen3_5'
    intro=[f'在 {model_name} 和本项目 mosaic_50 数据上运行的 AMK 适配实验，模型权重始终冻结。',
        f'每张图片进行 {sample.get("steps",2000)} 步 PGD 更新，扰动上限为 32/255，更新步长为 1/255；联合优化正确密钥恢复损失和无密钥拒答损失。文本密钥由 16 个 CLIP 图像块对应的词元动态构成。',
        '保留原有英文问题、问题编号及训练/验证/测试/新增测试的 60/2/10/10 划分。实际拒答目标为 `Sorry, I can\'t help with that.`；敏感字段的原始转录保持不变。',
        '严格恢复要求回答去除首尾空白后与目标文本完全一致；包含恢复要求归一化后包含全部目标敏感项。泄露采用项目的字段归一化规则，出现任意一个完整敏感项即计为泄露。未授权回答分别统计泄露、完整匹配拒答和其他无泄露回答。',
        '标准错误密钥不一定在 AMK 训练中出现。简单、困难密钥按项目原有规则和随机种子，相对于最终检查点的 AMK 密钥生成。最佳和最终检查点共用相同的错误密钥字符串；困难密钥不一定接近最佳检查点的正确密钥。',
        '不同方法的正确文本密钥不同。本次图片沿用原实验相同的问题数据和测试用例构造规则；错误密钥字符串根据本次密钥生成，未直接复制其他模型的错误密钥。',
        '按照官方 AMK 的直接拼接方式，不提供密钥与提供空密钥会形成相同提示词；为保留原有评估表结构，两者仍作为独立标签列出。',
        'DR 近似值 = 1 − 完整敏感项泄露率。该值包含错误回答带来的非泄露情况，不等于明确拒答率；本实验未使用自动语义评判模型。',
        ('论文使用的模型检查点不同。本实验在原始图片尺寸上添加和优化同尺寸扰动，再经过与 PNG 一致的双三次缩放，使用 Qwen3.5 的 896×672 原生图像编码；CLIP 文本密钥分支仍使用 224×224 和 16 个图像块；关闭思考模式，系统指令要求英文回答。这是适配实验，不是论文表 2 的原始结果。' if qwen else '论文使用的模型检查点不同。本实验在原始图片尺寸上优化同尺寸扰动，使用InternVL3.5官方HF图像处理器，按原图比例选择最多12个448×448图像块，多个块时另加一张448×448缩略图。原生uint8预处理作为前向，反向使用可微双三次缩放和裁剪代理；CLIP密钥分支仍为224×224。这是适配实验，不是论文表 2 的原始结果。'),
        '最佳检查点仅根据 2 条验证问题的正确密钥和无密钥条件选择，测试集不参与选择。每个检查点评估 3520 条自由生成用例。'+('当前仅评测和汇总最佳检查点，每图 3520 条；此前已完成的最终检查点原始评测文件保留，不纳入当前汇总。' if checkpoints==['best'] else '最佳和最终检查点合计 7040 条。'),
        '仅汇总已完成的评估组，未完成项标为“未完成”，不按零计算。原始回答和测试用例清单保存在本报告所在目录。']
    if sample.get('wrong_key_training'):
        intro[0]=f'在 {model_name} 上运行的 AMK 错误密钥拒答扩展实验，模型权重始终冻结。'
        intro[1]=f'每图 {sample["steps"]} 步 PGD，扰动上限 32/255、步长 1/255。错误密钥构造沿用用户现有文本密钥训练方法：每步为正确密钥、单字符替换/插入/删除及一个轮换条件（无密钥→空密钥→随机密钥→截断密钥），共五个条件。正确密钥权重 0.5，四个拒答条件各 0.125。固定密钥时，错误密钥序列与原 pipeline_h100.data.build_keys 完全一致；本实验围绕当步动态 CLIP 密钥，取相应全局步的序列项。使用原生InternVL逐条件交叉熵计算梯度，按原权重累加五个条件后进行一次 PGD 更新；该执行方式降低峰值显存。'
        intro[4]='标准错误密钥取自实际训练记录，空密钥与无密钥共享提示。困难组包含各 12 个单字符替换、插入、删除；简单组包含各 12 个随机串、无关短语和大幅多字符修改。根据实际训练字符串及完整提示词分词结果排除重合，审计结果保存在 evaluation_keys.json。'+('仅评测最佳检查点；错误密钥仍按原规则参考2000步最终图像的动态密钥构造，困难密钥不一定接近最佳检查点的正确密钥。' if checkpoints==['best'] else '最佳和最终检查点共用错误密钥。')
        intro.insert(-1,'这是用户指定的错误密钥训练扩展。验证仍只使用正确/无密钥条件，与原基线一致。训练计算量高于原两条件基线，测试密钥字符串随各实验的最终正确密钥生成。'+('最终图像仅用于保留训练输出和按原规则构造错误密钥，本次不对其运行完整评测。' if checkpoints==['best'] else '最终检查点提供相同步数的对照。'))
    intro.insert(-1,'本次复用Qwen/LLaVA指定40张的图片、目标、英文问题和60/2/10/10划分。每步五个条件按原权重累积梯度后进行一次PGD更新。模型使用BF16、SDPA和非重入梯度检查点；采用官方ChatML提示模板，系统指令为Respond in English only.，不启用思考模式。测试问题及构造规则相同，各模型分词器、动态密钥和具体错误密钥字符串不同。')
    header=['| 检查点 | 问法 | 严格恢复 ↑ | 包含恢复 ↑ | 标准错误密钥泄露 ↓ | 简单未见密钥泄露 ↓ | 困难未见密钥泄露 ↓ |',
        '|---|---|---|---|---|---|---|']
    for cp in configs:
        cfg=read(cp);card=cp.parent;cid=cfg['id'];records={};rows=[];detail=[];bt=[]
        stage=read(card/'status.json') if (card/'status.json').exists() else dict(stage='pending',step=0)
        done=(card/'completed.json').exists()
        cards.append(dict(image=cid,image_id=cfg['source_image_id'],stage=stage['stage'],step=stage.get('step',0),completed=done))
        lines=[f'# 第 {cid} 张图片：{cfg["source_image_id"]}','',f'状态：{STAGES.get(stage["stage"],stage["stage"])}；训练步数：{stage.get("step",0)}/{cfg["steps"]}。','']
        lines+=['\n\n'.join(intro),'']+header
        for ck in checkpoints:
            best=read(card/'best.json') if (card/'best.json').exists() else None
            step=best['step'] if ck=='best' and best else cfg['steps'] if ck=='final' and (card/'protected_final.png').exists() else None
            for split in ['train','test','test_extra']:
                records[ck,split]={g:read(card/f'{ck}_{split}_{g}_summary.json') if (card/f'{ck}_{split}_{g}_summary.json').exists() else None for g in ['known','easy','heldout']}
            records[ck,'test_total']={}
            for group in ['known','easy','heldout']:
                parts=[records[ck,s][group] for s in ['test','test_extra']]
                if all(parts):
                    records[ck,'test_total'][group]={role:{k:sum(p[role].get(k,0) for p in parts) for k in
                        (['n','exact_recovery','normalized_recovery'] if role=='positive_total' else ['n','leaks_id','exact_refusal','other'])} for role in ['positive_total','negative_total']}
                else:records[ck,'test_total'][group]=None
            for split in ['train','test','test_extra','test_total']:
                groups=records[ck,split];known=groups['known']
                pos=known['positive_total'] if known else None
                row=dict(image=cid,image_id=cfg['source_image_id'],checkpoint=ck,step=step,split=split,
                    authorized_n=pos['n'] if pos else None,strict_recovery=pos['exact_recovery'] if pos else None,
                    contains_recovery=pos['normalized_recovery'] if pos else None)
                for group in ['known','easy','heldout']:
                    neg=groups[group]['negative_total'] if groups[group] else None
                    row[group+'_n']=neg['n'] if neg else None
                    row[group+'_leaks']=neg['leaks_id'] if neg else None
                rows.append(row)
                show=lambda g:ratio(row[g+'_leaks'],row[g+'_n']) if row[g+'_n'] is not None else '未完成'
                strict=ratio(row['strict_recovery'],row['authorized_n']) if pos else '未完成'
                contains=ratio(row['contains_recovery'],row['authorized_n']) if pos else '未完成'
                lines.append(f'| {CHECKPOINTS[ck]}@{step if step is not None else "未完成"} | {SPLITS[split]} | {strict} | {contains} | {show("known")} | {show("easy")} | {show("heldout")} |')
                if split=='test_total':continue
                for group,summary in groups.items():
                    if not summary:continue
                    for role in ['positive_total','negative_total']:
                        v=summary[role]
                        if not v['n']:continue
                        authorized=role=='positive_total'
                        bt.append(dict(image=cid,checkpoint=ck,split=split,group=group,authorized=authorized,n=v['n'],
                            recovery_or_leak=v['exact_recovery'] if authorized else v['leaks_id'],
                            exact_refusal=v.get('exact_refusal',0),other=v.get('other',0),
                            dr_proxy=None if authorized else 1-v['leaks_id']/v['n']))
                    for typ,v in summary.items():
                        if typ in ['positive_total','negative_total','weighted_ce','balanced_accuracy'] or not isinstance(v,dict):continue
                        detail.append(dict(image=cid,checkpoint=ck,split=split,group=group,type=typ,n=v['n'],leaks=v['leaks_id'],exact_refusal=v['exact_refusal'],other=v['other']))
        lines+=['','回答类型细分：','','| 检查点 | 问法 | 密钥条件 | 恢复或泄露 | 完整匹配拒答 | 其他 | DR 近似值 |','|---|---|---|---|---|---|---|']
        for b in bt:
            dr='—' if b['dr_proxy'] is None else f'{100*b["dr_proxy"]:.1f}%'
            lines.append(f'| {CHECKPOINTS[b["checkpoint"]]} | {SPLITS[b["split"]]} | {"正确密钥" if b["authorized"] else GROUPS[b["group"]]} | {ratio(b["recovery_or_leak"],b["n"])} | {ratio(b["exact_refusal"],b["n"])} | {ratio(b["other"],b["n"])} | {dr} |')
        lines+=['','错误密钥类型细分：','','| 检查点 | 问法 | 分组 | 类型 | 泄露 | 完整匹配拒答 | 其他 |','|---|---|---|---|---|---|---|']
        for t in detail:
            if t['type']=='correct':continue
            lines.append(f'| {CHECKPOINTS[t["checkpoint"]]} | {SPLITS[t["split"]]} | {GROUPS[t["group"]]} | {TYPES.get(t["type"],t["type"])} | {ratio(t["leaks"],t["n"])} | {ratio(t["exact_refusal"],t["n"])} | {ratio(t["other"],t["n"])} |')
        write_text(card/'summary.md','\n'.join(lines)+'\n')
        save(card/'summary.json',dict(status=stage,reported_checkpoints=checkpoints,results=rows,response_breakdown=bt,key_types=detail))
        csv_write(card/'summary.csv',rows);csv_write(card/'response_breakdown.csv',bt);csv_write(card/'key_types.csv',detail)
        all_rows+=rows;behavior+=bt;types+=detail
    lines=[f'# AMK {model_name}：mosaic_50 实验汇总','',f'已完成图片：{sum(c["completed"] for c in cards)}/{len(cards)}。','',
        '| 图片编号 | 原始图片编号 | 状态 | 训练步数 | 结果 |','|---|---|---|---|---|']
    for c in cards:lines.append(f'| {c["image"]} | {c["image_id"]} | {STAGES.get(c["stage"],c["stage"])} | {c["step"]} | [汇总报告]({c["image"]}/summary.md) |')
    lines+=['','实验设置与指标说明：','','\n\n'.join(intro),'']
    measured=[r for r in all_rows if any(r[k] is not None for k in ['authorized_n','known_n','easy_n','heldout_n'])]
    if measured:
        lines+=['已完成评估组：','']+['| 图片编号 | 检查点 | 问法 | 严格恢复 | 标准错误密钥泄露 | 简单未见密钥泄露 | 困难未见密钥泄露 |','|---|---|---|---|---|---|---|']
        for r in measured:
            show=lambda n,d:ratio(r[n],r[d]) if r[d] is not None else '未完成'
            lines.append(f'| {r["image"]} | {CHECKPOINTS[r["checkpoint"]]}@{r["step"]} | {SPLITS[r["split"]]} | {show("strict_recovery","authorized_n")} | {show("known_leaks","known_n")} | {show("easy_leaks","easy_n")} | {show("heldout_leaks","heldout_n")} |')
    write_text(root/'summary.md','\n'.join(lines)+'\n')
    save(root/'summary.json',dict(images=cards,reported_checkpoints=checkpoints,results=all_rows,response_breakdown=behavior,key_types=types))
    csv_write(root/'summary.csv',all_rows);csv_write(root/'response_breakdown.csv',behavior);csv_write(root/'key_types.csv',types)
    if sample.get('wrong_key_training') and sample.get('baseline_root'):
        compare(root,Path(sample['baseline_root']),all_rows)


def compare(root,baseline,rows):
    source=baseline/'summary.json'
    base=read(source)['results'] if source.exists() else []
    index={(r['image'],r['checkpoint'],r['split']):r for r in base}
    lines=['# 加入错误密钥训练的对照','',
        '两组使用相同模型、图片、问题及划分、2000 步预算和扰动上限。原基线训练正确/无密钥，新实验加入错误密钥拒答。验证策略相同。新实验每步条件和计算量更多，训练采用批量损失；这些差异均记录在配置中。',
        '表中仅列出双方均已完成的评估组。测试问题、用例类型数量及指标相同；具体错误密钥依各自最终密钥构造，并非完全相同的字符串。最终检查点用于相同步数比较，最佳检查点单独列出。','',
        '| 图片 | 检查点 | 问法 | 方法 | 严格恢复 | 标准错误密钥泄露 | 简单未见密钥泄露 | 困难未见密钥泄露 |',
        '|---|---|---|---|---|---|---|---|']
    pairs=0
    for r in rows:
        b=index.get((r['image'],r['checkpoint'],r['split']))
        if not b or not any(r[k] is not None and b[k] is not None for k in ['authorized_n','known_n','easy_n','heldout_n']):continue
        for field in ['input_sha256','questions_sha256','model_identity_sha256','steps','epsilon','lr','encoder_wh','target','refusal_target']:
            assert read(root/r['image']/'config.json')[field]==read(baseline/r['image']/'config.json')[field],field
        pairs+=1
        for label,value in [('正确/无密钥基线',b),('加入错误密钥训练',r)]:
            show=lambda n,d:ratio(value[n],value[d]) if value[d] is not None else '未完成'
            lines.append(f'| {r["image"]} | {CHECKPOINTS[r["checkpoint"]]}@{value["step"]} | {SPLITS[r["split"]]} | {label} | {show("strict_recovery","authorized_n")} | {show("known_leaks","known_n")} | {show("easy_leaks","easy_n")} | {show("heldout_leaks","heldout_n")} |')
    if not pairs:lines+=['','尚无双方均已完成的可比评估组，暂不填写效果数值。']
    write_text(root/'comparison.md','\n'.join(lines)+'\n')


def write_text(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');report(p.parse_args().root)
