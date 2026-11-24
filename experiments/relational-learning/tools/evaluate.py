#!/usr/bin/env python3
"""Counterfactual preference scoring: base vs trained; full vs final-message-only.

Scores likelihood of the SAME response candidates under each context. This is a
behavioral diagnostic, not an empathy score or a human judgment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from review import load_jsonl, save_jsonl
from validate import validate

SCORING = ('base_full', 'base_last', 'trained_full', 'trained_last')


def candidate_pair(row: dict) -> tuple[str, str]:
    """Get stable A/B candidate order across both counterfactual variants."""
    return ((row['chosen'], row['rejected']) if row['variant'] == 1
            else (row['rejected'], row['chosen']))


def load_evaluation(sources: list[Path]) -> list[dict]:
    errs = validate([str(x) for x in sources])
    if errs:
        raise ValueError('Invalid evaluation data: ' + '; '.join(errs))
    rows = [r for p in sources for r in load_jsonl(p)]
    if any(r['split'] not in {'dev','test'} for r in rows):
        raise ValueError('Training-split rows cannot be used as held-out evaluation')
    return rows


def metrics(rows: list[dict]) -> dict:
    """Compute context reversal accuracies from precomputed score margins."""
    families = defaultdict(list)
    for row in rows:
        families[row['family_id']].append(row)
    for key, vals in families.items():
        if len(vals)!=2 or {r['variant'] for r in vals}!={1,2}:
            raise ValueError(f'Missing or duplicate paired variants in {key}')
    output = {'families':len(families),'pairs':len(rows)}
    for name in SCORING:
        good_rows=good_families=0
        for two in families.values():
            scored=[]
            for row in two:
                diff=row[name]
                if type(diff) not in (int,float) or not (-1e10 < diff < 1e10):
                    raise ValueError(f'Invalid score for {name}')
                correct = diff > 0 if row['variant']==1 else diff < 0
                good_rows+=int(correct)
                scored.append(correct)
            good_families += int(all(scored))
        output[name]={'row_correct':good_rows,'row_total':len(rows),
                      'row_accuracy':good_rows/len(rows),
                      'family_flip_correct':good_families,
                      'family_total':len(families),
                      'family_flip_accuracy':good_families/len(families)}
    return output


def summarize(scores: Path, report: Path) -> dict:
    if report.exists() or report.resolve()==scores.resolve():
        raise ValueError('Report must be new and distinct from source')
    rows=load_jsonl(scores)
    seen=set()
    for row in rows:
        if set(row)!=({'id','family_id','variant','split'}|set(SCORING)):
            raise ValueError('Unknown or missing score fields')
        if row['id'] in seen: raise ValueError('Duplicate scored row')
        seen.add(row['id'])
        if row['split'] not in {'dev','test'}: raise ValueError('Non-held-out scored split')
    summary=metrics(rows)
    summary.update({'score_sha256':hashlib.sha256(scores.read_bytes()).hexdigest(),
      'warning':'Synthetic author-label accuracy is not a measure of human empathy; public test cases are not contamination-resistant.'})
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary


def score_response(model, tokenizer, context: list[dict], completion: str) -> float:
    """Per-token conditional log likelihood, without silently truncating context."""
    import torch
    prompt=tokenizer.apply_chat_template(context,tokenize=True,add_generation_prompt=True)
    if isinstance(prompt,dict):prompt=prompt['input_ids']
    reply=tokenizer(completion,add_special_tokens=False)['input_ids']
    if not prompt or not reply:
        raise ValueError('Empty prompt or completion after tokenization')
    reply=reply+[tokenizer.eos_token_id]
    sequence=prompt+reply
    limit=getattr(model.config,'max_position_embeddings',None)
    if limit is not None and len(sequence)>limit:
        raise ValueError('Input exceeds model context; do not silently truncate evaluation')
    device=next(model.parameters()).device
    tokens=torch.tensor([sequence],dtype=torch.long,device=device)
    with torch.inference_mode():
        logits=model(input_ids=tokens).logits[0,len(prompt)-1:-1]
        targets=tokens[0,len(prompt):]
        logp=logits.float().log_softmax(dim=-1).gather(1,targets[:,None]).squeeze(1)
        return float(logp.mean().item())


def score(args) -> dict:
    if args.out.exists() or args.out.resolve() in {p.resolve() for p in args.data}:
        raise ValueError('Score output must be new and distinct from inputs')
    rows=load_evaluation(args.data)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError('Scoring requires torch, transformers and peft') from exc
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForCausalLM.from_pretrained(args.model, dtype='auto')
    if torch.cuda.is_available(): model=model.to('cuda')
    model.eval()
    results=[{'id':r['id'],'family_id':r['family_id'],'variant':r['variant'],'split':r['split']}
             for r in rows]
    def evaluate_variant(model, prefix: str):
        for source,dest in zip(rows,results):
            a,b=candidate_pair(source)
            for mode in ['full','last']:
                context=source['context'] if mode=='full' else [source['context'][-1]]
                # Each family has exactly the same two responses in both contexts.
                dest[prefix+'_'+mode]=score_response(model,tokenizer,context,a)-score_response(model,tokenizer,context,b)
    evaluate_variant(model,'base')
    if args.adapter:
        model=PeftModel.from_pretrained(model,args.adapter)
        model.eval()
        evaluate_variant(model,'trained')
    else:
        for x in results:
            x['trained_full']=x['base_full'];x['trained_last']=x['base_last']
    save_jsonl(args.out,results)
    return {'pairs':len(results),'families':len({r['family_id'] for r in rows}),
            'model':args.model,'adapter':args.adapter or None,
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data},
            'scores_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
            'warning':'No adapter means identity sanity check, not a trained-model comparison.'}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    s=commands.add_parser('score')
    s.add_argument('--data',type=Path,nargs='+',required=True)
    s.add_argument('--model',default='Qwen/Qwen2.5-0.5B-Instruct')
    s.add_argument('--adapter',type=str)
    s.add_argument('--out',type=Path,required=True)
    q=commands.add_parser('summarize')
    q.add_argument('--scores',type=Path,required=True)
    q.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=score(args) if args.command=='score' else summarize(args.scores,args.report)
        print(json.dumps(result,indent=2))
    except (OSError,ValueError,RuntimeError,TypeError) as exc:
        parser.error(str(exc))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
