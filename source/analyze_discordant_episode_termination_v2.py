"""Audit both directions of paired success differences from frozen records/logs."""
from pathlib import Path
import argparse,ast,collections,concurrent.futures,hashlib,json,math,re,time
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root)
O=R/'analysis/discordant_episode_termination_002';O.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=R/'analysis/deep_policy_trajectory_diagnosis_001/episode_diagnostics.json'
rows=read(source);assert len(rows)==5184
pairs=collections.defaultdict(dict)
for r in rows:pairs[(r['split'],r['seed'],r['episode_id'])][r['group']]=r
discordant=[];targets={}
for (split,seed,eid),g in pairs.items():
    gt,rec=g['gt-joint-boundary'],g['rec-joint-boundary']
    if gt['success']==rec['success']:continue
    winner='REC' if rec['success'] else 'GT'
    for name,row in [('GT',gt),('REC',rec)]:
        key=(split,seed,eid,name);targets[key]=row
    loser=gt if winner=='REC' else rec
    discordant.append({'split':split,'seed':seed,'episode_id':eid,'winner':winner,
                       'failure_category':('ended_inside_but_official_failure' if loser['final_XY_distance']<=20 else 'reached_then_ended_outside') if loser['oracle_success'] else 'never_reached',
                       'winner_key':(split,seed,eid,winner),'loser_key':(split,seed,eid,'GT' if winner=='REC' else 'REC')})
assert len(discordant)==257 and len(targets)==514
official=R.parent.parent/'aeroact/20261018-012530-nvila8b-aerialvln-s-official/eval/official_annotations'
ann={s:{str(e['episode_id']):e for e in read(official/(s+'.json'))['episodes']} for s in ['val_seen','val_unseen']}
S=R/'source/release-collator-f9b611eb926ef29d/eval.py'
fresh=read(R/'analysis/official_GT_external_recheck_001/receipt.json')
assert sha(S)==next(x['sha256'] for x in fresh['sources'] if x['path']=='eval.py')
fn=next(n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='convert_text_to_action')
ns={'re':re,'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(S),'exec'),ns)
parse=ns['convert_text_to_action']
def load(item):
    key,row=item;p=R/row['record_source'];assert sha(p)==row['episode_sha256'];ep=read(p)
    log=p.parent.parent/'client.log';assert log.exists()
    return key,row,ep,log
loaded=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:loaded=list(pool.map(load,targets.items()))
bylog=collections.defaultdict(list)
for entry in loaded:bylog[entry[3]].append(entry)
number=r'[-+0-9.eE]+'
result=re.compile(r'result-0\s+distance_to_goal:\s*('+number+r')\s+success:\s*('+number+r')\s+ndtw:\s*('+number+r')\s+sdtw:\s*('+number+r')\s+path_length:\s*('+number+r')\s+oracle_success:\s*('+number+r')\s+steps_taken:\s*('+number+r')')
calls=re.compile(r'\bstep (\d+) action_text: (.*?) actions: (\[[^\]\r\n]*\])',re.S)
info=re.compile(r'(?m)^\s*(\d+) (True|False) ('+number+r') ('+number+r') ('+number+r') (True|False)\s*$')
forces=re.compile(r'step (\d+) force stop the episode[^\r\n]*')
fields=['distance_to_goal','success','ndtw','sdtw','path_length','oracle_success','steps_taken']
checked={};unresolved=[];logproof=[]
def equal_metric(vals,ep):return all(math.isclose(float(v),float(ep[k]),rel_tol=1e-8,abs_tol=1e-7) for k,v in zip(fields,vals))
for lp,rr in bylog.items():
    b=lp.read_bytes();text=b.decode(errors='replace');matches=list(result.finditer(text));begin=0;found=set()
    for m in matches:
        block=text[begin:m.end()];begin=m.end();candidates=[]
        for key,row,ep,_ in rr:
            if key in found or not equal_metric(m.groups(),ep):continue
            instruction=ann[key[0]][key[2]]['instruction']['instruction_text']
            if instruction in block:candidates.append((key,row,ep))
        if not candidates:continue
        if len(candidates)!=1:
            unresolved.append({'log':str(lp.relative_to(R)),'reason':'Ambiguous identical instruction/metric signatures','keys':[list(x[0]) for x in candidates]});continue
        key,row,ep=candidates[0];cc=list(calls.finditer(block));ii=list(info.finditer(block))
        assert cc and ii,(key,'Missing call or step log')
        flat=[]
        for cm in cc:
            acts=ast.literal_eval(cm.group(3));assert parse(cm.group(2))==acts,(key,cm.group(2));flat.extend(acts)
        assert flat==ep['action_record'],(key,'Proposed actions differ from episode record')
        cm=cc[-1];im=ii[-1];loop_step=int(im.group(1));call_step=int(cm.group(1));actions=ast.literal_eval(cm.group(3));j=loop_step-call_step-1
        assert 0<=j<len(actions),(key,loop_step,call_step,actions)
        attempted=actions[j];assert im.group(2)=='True' and math.isclose(float(im.group(3)),float(ep['distance_to_goal']),abs_tol=1e-7)
        ff=[f.group(0) for f in forces.finditer(block) if int(f.group(1))==loop_step]
        collision=im.group(6)=='True'
        if ff:trigger='repetition_guard_forced_Stop'
        elif attempted==0:trigger='model_text_parsed_Stop'
        elif ep['steps_taken']>=500:trigger='action_budget_termination'
        elif collision:trigger='depth_proxy_termination'
        else:trigger='unresolved'
        checked[key]={'split':key[0],'seed':key[1],'episode_id':key[2],'policy':key[3],
          'success':row['success'],'ever_within_20m':row['oracle_success'],'final_XY_distance':row['final_XY_distance'],
          'minimum_XY_distance':row['minimum_XY_distance'],'termination_trigger':trigger,
          'final_model_text':cm.group(2),'final_parsed_actions':actions,'attempted_action_at_ending_step':attempted,
          'final_loop_step':loop_step,'final_call_start_step':call_step,'forced_guard_messages':ff,
          'depth_proxy_flag_at_terminal_observation':collision,'post_first_entry_transitions':row['post_first_entry_transitions'],
          'log_relative':str(lp.relative_to(R)),'log_sha256':hashlib.sha256(b).hexdigest(),
          'terminal_log_excerpt':block[max(cm.start(),len(block)-2000):],
          'episode_source':row['record_source'],'episode_sha256':row['episode_sha256']}
        found.add(key)
    missing=[list(key) for key,_,_,_ in rr if key not in found]
    if missing:unresolved.append({'log':str(lp.relative_to(R)),'reason':'No unique complete block matched','keys':missing})
    logproof.append({'log':str(lp.relative_to(R)),'sha256':hashlib.sha256(b).hexdigest(),'target_episodes':len(rr),'matched_target_episodes':len(found),'all_result_blocks':len(matches)})
assert len(checked)==514 and not unresolved,(len(checked),unresolved[:5])
assert not any(v['termination_trigger']=='unresolved' for v in checked.values())
def summarize(pp):
    out={}
    for win in ['REC','GT']:
        cases=[x for x in pp if x['winner']==win];group={}
        for category in ['reached_then_ended_outside','ended_inside_but_official_failure','never_reached']:
            ss=[x for x in cases if x['failure_category']==category]
            group[category]={'pairs':len(ss),'winner_terminal_triggers':dict(collections.Counter(checked[x['winner_key']]['termination_trigger'] for x in ss)),
                            'loser_terminal_triggers':dict(collections.Counter(checked[x['loser_key']]['termination_trigger'] for x in ss))}
        out[win+'_only_success']={'pairs':len(cases),'loser_reached_then_failed':group['reached_then_ended_outside']['pairs']+group['ended_inside_but_official_failure']['pairs'],
              'loser_never_reached':group['never_reached']['pairs'],'categories':group,
              'winner_terminal_triggers':dict(collections.Counter(checked[x['winner_key']]['termination_trigger'] for x in cases)),
              'loser_terminal_triggers':dict(collections.Counter(checked[x['loser_key']]['termination_trigger'] for x in cases))}
    return out
pooled={s:summarize([x for x in discordant if x['split']==s]) for s in ['val_seen','val_unseen']}
perseed={str(seed):{s:summarize([x for x in discordant if x['split']==s and x['seed']==seed]) for s in pooled} for seed in [10,11,12]}
decomp={}
for s,v in pooled.items():
    a,b=v['REC_only_success'],v['GT_only_success']
    decomp[s]={'net_REC_success_gain':a['pairs']-b['pairs'],'net_gain_in_both_reached_pairs':a['loser_reached_then_failed']-b['loser_reached_then_failed'],
               'net_gain_in_loser_never_reached_pairs':a['loser_never_reached']-b['loser_never_reached'], 'net_gain_in_ended_outside_after_reaching':a['categories']['reached_then_ended_outside']['pairs']-b['categories']['reached_then_ended_outside']['pairs'], 'net_gain_in_ended_inside_but_failed':a['categories']['ended_inside_but_official_failure']['pairs']-b['categories']['ended_inside_but_official_failure']['pairs']}
result={'created_at':time.time(),'discordant_pairs':len(discordant),'audited_policy_runs':len(checked),'source_logs':len(logproof),
 'all_ending_triggers_resolved_from_logs':True,'pooled_by_split':pooled,'per_seed':perseed,'descriptive_net_gain_decomposition':decomp,
 'original_episode_diagnostics_sha256':sha(source),'official_parser_source_sha256':sha(S),'source_sha256':sha(Path(__file__)),
 'limits':['Counts pool repeated instructions across three training seeds; they are runs, not unique tasks or independent trials.',
           'Reached then failed describes an outcome trajectory, not proof of a causal late-Stop defect. Expert trajectories can legitimately move after entering20m.',
           'Model-text Stop means the released parser maps a generated text to Stop; this does not prove correct instruction progress understanding.',
           'Repetition-guard terminations and depth-proxy terminations must not be reported as voluntary model Stop.',
           'Same episode IDs do not imply the two policies saw the same states. Group differences are descriptive, not a causal mediation estimate.',
           'Depth-proxy flags may co-occur with an earlier Stop trigger; terminal trigger classification follows execution order, not the final flag alone.'],
 'new_model_training_simulator_or_navigation_run':False}
paired=[]
for x in discordant:
    paired.append({k:v for k,v in x.items() if not k.endswith('_key')}|{'winner_run':checked[x['winner_key']],'loser_run':checked[x['loser_key']]})
for name,value in [('summary.json',result),('paired_records.json',paired),('log_provenance.json',logproof)]:
    (O/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(O/'exit.json').write_text(json.dumps({'success':True,'source_sha256':sha(Path(__file__)),
 'files_sha256':{p.name:sha(p) for p in O.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'output':str(O),'pooled_by_split':pooled,'descriptive_net_gain_decomposition':decomp},ensure_ascii=False),flush=True)
