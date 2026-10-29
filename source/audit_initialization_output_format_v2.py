"""Read-only live/finished shard audit; never alters policy or navigation metrics."""
from pathlib import Path
import argparse, ast, collections, hashlib, json, math, re, time
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
a=ap.parse_args();R=Path(a.root).resolve()
E=R/'eval/initialization_policy_control_002'
source=Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source'])/'eval.py'
text=source.read_text(errors='replace');tree=ast.parse(text)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='convert_text_to_action')
namespace={'re':re,'math':math}
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
parser=namespace['convert_text_to_action']
pattern_assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='patterns' for x in n.targets))
exec(compile(ast.Module(body=[pattern_assignment],type_ignores=[]),str(source),'exec'),namespace)
patterns=namespace['patterns']
pattern=re.compile(r'^step (\d+) action_text: (.*?) actions: (\[[^\]\n]*\])\r?$',re.M|re.S)
def state(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.read_text().split(') ',1)[1].split()[0] if p.exists() else 'MISSING'
plan=json.loads((E/'plan.json').read_text())
assigned={s:set(ids) for s,ids in plan['episode_ids'].items()}
workers=[];count=collections.Counter();all_examples=[];all_ids=collections.defaultdict(set)
for procfile in sorted((E/'base/attempt-001').glob('shard-*/process.json')):
    p=json.loads(procfile.read_text());D=procfile.parent
    log=(D/'client.log').read_text(errors='replace') if (D/'client.log').exists() else ''
    c=collections.Counter();examples=[];last_step=None
    for match in pattern.finditer(log):
        step,output,reported=match.groups();last_step=int(step)
        reported=ast.literal_eval(reported);actual=parser(output)
        c['observed_generation_calls']+=1
        c['reported_parser_mismatches']+=int(reported!=actual)
        matched=next((act for act,rx in patterns.items() if rx.search(output)),None)
        if matched is None and re.search(r'\d+',output):kind='unmatched_text_digit_triggers_default_right'
        elif matched is None:kind='unmatched_text_returns_None'
        elif matched==0:kind='stop_keyword_recognized'
        else:kind='motion_pattern_recognized'
        c[kind]+=1
        if len(examples)<3 and kind.startswith('unmatched'):
            examples.append({'step':int(step),'text':output,'reported_actions':reported,'diagnostic_kind':kind})
    W=Path(p['workspace'])
    records={};pending_writes=0;copied=0;live=0;duplicate_agreements=0
    candidates=[('copied', x) for x in sorted((D/'episodes').glob('*.json'))]
    candidates += [('live', x) for x in sorted((W/'code/AirVLN_eval').glob('**/intermediate_results_every/*.json'))]
    for origin,candidate in candidates:
        try:record=json.loads(candidate.read_text())
        except json.JSONDecodeError:
            pending_writes+=1
            continue
        assert record['success'] in [0,1] and record['oracle_success'] in [0,1]
        assert all(math.isfinite(float(record[k])) for k in ['success','oracle_success','distance_to_goal','ndtw','sdtw','steps_taken'])
        eid=candidate.stem
        if eid in records:
            assert records[eid]==record, ('Copied/live episode disagreement',eid,str(candidate))
            duplicate_agreements+=1
        else:records[eid]=record
        copied+=int(origin=='copied');live+=int(origin=='live')
    ids=set(records)
    terminal=json.loads((D/'exit.json').read_text()) if (D/'exit.json').exists() else None
    if terminal is not None and terminal['success']:
        assert copied==len(p['assigned_episode_ids']) and ids==set(p['assigned_episode_ids'])
        assert terminal['client_returncode']==0
    assert ids<=set(p['assigned_episode_ids'])
    assert ids<=assigned[p['split']] and not (all_ids[p['split']]&ids)
    all_ids[p['split']].update(ids)
    count.update(c);all_examples.extend(examples)
    workers.append({'worker':D.name,'gpu':p['gpu'],'split':p['split'],
        'worker_pid':p['pid'],'worker_process_state':state(p['pid']),
        'client_pid':p['client_pid'],'client_process_state':state(p['client_pid']) if p['client_pid'] else None,
        'completed_episode_records':len(records),'copied_records':copied,'live_records':live,'duplicate_copied_live_agreements':duplicate_agreements,'terminal_receipt':terminal,'pending_record_writes':pending_writes,'assigned_episodes':len(p['assigned_episode_ids']),
        'last_observed_step':last_step,'log_mtime':(D/'client.log').stat().st_mtime if (D/'client.log').exists() else None,
        'format_counts':dict(c),'examples':examples})
assert count['reported_parser_mismatches']==0
result={'scope':'Live snapshot of initialization-only control. Counts may include incomplete episodes and are not final metrics.',
 'created_at':time.time(),'eval_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'parser_function_AST_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
 'controller_pid':json.loads((E/'launch.json').read_text())['pid'],'controller_state':state(json.loads((E/'launch.json').read_text())['pid']),
 'workers':workers,'format_counts':dict(count),'completed_IDs':{s:sorted(ids) for s,ids in all_ids.items()},
 'completed_by_split':{s:len(ids) for s,ids in all_ids.items()},
 'controller_terminal_receipt':json.loads((E/'controller_exit.json').read_text()) if (E/'controller_exit.json').exists() else None,
 'format_failure_fraction':(count['unmatched_text_returns_None']+count['unmatched_text_digit_triggers_default_right'])/count['observed_generation_calls'] if count['observed_generation_calls'] else None,
 'official_parser_unchanged':True,'training_or_policy_modified':False,
 'interpretation':'Base-vs-finetuned utility combines action-format compliance with navigation. It cannot by itself isolate geometric pseudo-label usefulness.'}
p=R/'report/initialization_output_format_snapshot_v2.json'
q=p.with_suffix('.json.partial');q.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
print(json.dumps({k:v for k,v in result.items() if k not in ['workers','completed_IDs']},indent=2))
for w in workers:print('WORKER',w['worker'],w['completed_episode_records'],w['format_counts'])
