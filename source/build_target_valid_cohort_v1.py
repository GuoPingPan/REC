"""Create an immutable paired training view excluding non-retained text targets.

Keep parent cohorts, original RGB, actions, instructions and held-out splits.
No action relabeling, text rewriting, replay, model training or cloud submission.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,os,time
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args()
R=Path(a.root).resolve();P=R/'dataset/complete-optical-screened-metric-v3';C=R/'dataset/complete-optical-target-valid-v4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
assert read(R/'pipeline/optical_screened_dataset_v3_005/exit.json')['success']
assert read(R/'pipeline/verified_data_analysis_002/exit.json')['success']
V=read(P/'validation_report.json');ready=read(P/'READY.json');assert sha(P/'validation_report.json')==ready['validation_report_sha256']
assert (P/'TRAINING_USE_HOLD.json').exists()
audit_path=R/'eval/multimodal_target_retention_002/summary.json';audit=read(audit_path)
assert read(R/'report/multimodal_target_retention_002_exit.json')['returncode']==0
assert audit['dataset_validation_sha256']==sha(P/'validation_report.json') and len(audit['cases'])==9
text_path=R/'report/instruction_target_truncation_v1.json';text=read(text_path)
assert text['dataset_validation_sha256']==sha(P/'validation_report.json')
bad={x['episode'] for x in audit['affected_STOP_instructions']};assert bad
surplus=audit['feature_surplus_per_fixed8frame_input'];assert surplus>=0
retained_text=[x for x in text['records'] if x['episode'] not in bad]
assert len(retained_text)+len(bad)==V['usable_instructions']
for x in retained_text:assert x['last_supervised_token_index']+surplus<4096
minimum_margin=min(4096-1-(x['last_supervised_token_index']+surplus) for x in retained_text)
responses={s for counts in V['action_text_counts'].values() for s in counts}
responses.add('I think I should stop because I have finished the instruction.')
assert all(s.isascii() for s in responses)
response_byte_bound=max(len(s.encode()) for s in responses)+32
assert minimum_margin>response_byte_bound, 'Review boundary cases using actual preprocessing before constructing a view.'
rows=read(P/'route_inventory.json');new_rows=[]
for x in rows:
 episodes=[e for e in x['episodes'] if e not in bad]
 assert episodes, ('Route would lose all instructions; revise geometry/coverage accounting',x['route_id'])
 new_rows.append({**x,'episodes':episodes,'parent_C3_episode_ids':x['episodes']})
assert len(new_rows)==V['usable_routes']==2264
ids={e for x in new_rows for e in x['episodes']};assert len(ids)==sum(len(x['episodes']) for x in new_rows)
assert ids=={x['episode'] for x in retained_text}
assert not C.exists();C.mkdir();save(C/'BUILDING.json',{'at':time.time(),'source_sha256':sha(Path(__file__)),'parent_validation_sha256':sha(P/'validation_report.json')})
def check_manifest(row):
 assert sha(Path(row['rgb'])/'manifest.json')==row['RGB_manifest_sha256']
 return row['frame_count']
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:frames=sum(pool.map(check_manifest,new_rows))
assert frames==V['physical_RGB_frames']
save(C/'route_inventory.json',new_rows)
save(C/'excluded_route_inventory.json',read(P/'excluded_route_inventory.json'))
save(C/'excluded_instruction_inventory.json',{'selection_rule':'Exclude every instruction whose STOP response loses any supervised token under released4096token collator plus observed fixed8frame multimodal expansion. No navigation outcomes used.','audit_sha256':sha(audit_path),'excluded':audit['affected_STOP_instructions']})
(C/'route_labels').symlink_to(os.path.relpath(P/'route_labels',C),target_is_directory=True)
counts={};classes={};digests={};joint={}
for group in V['sample_counts']:
 src=P/group/'data/aerialvln-s';dst=C/group/'data/aerialvln-s'
 for name,digest in V['annotation_sha256'][group].items():assert sha(src/name)==digest
 xs=[x for x in read(src/'train_merged_triple.json') if x['episode'] in ids]
 idx={k:v for k,v in read(src/'train_episodes2idx_merge.json').items() if k in ids}
 instructions={k:v for k,v in read(src/'train_episode2instruction.json').items() if k in ids}
 assert set(idx)==set(instructions)==ids
 stops=collections.Counter(x['episode'] for x in xs if x['action']=='The next action is stop.')
 assert set(stops)==ids and set(stops.values())=={1}
 for n,obj in [('train_merged_triple.json',xs),('train_episodes2idx_merge.json',idx),('train_episode2instruction.json',instructions),('train_action_prob_weight.json',read(src/'train_action_prob_weight.json'))]:save(dst/n,obj)
 rgb=C/group/'Raw_data/aerialvln-s';rgb.parent.mkdir(parents=True);rgb.symlink_to(os.path.relpath(P/group/'Raw_data/aerialvln-s',rgb.parent),target_is_directory=True)
 counts[group]=len(xs);classes[group]=dict(collections.Counter(x['action'] for x in xs));digests[group]={p.name:sha(p) for p in dst.glob('*.json')}
 if group.endswith('joint-boundary'):joint[group]=(xs,idx,instructions)
l,r=joint['gt-joint-boundary'],joint['rec-joint-boundary'];assert l[1:]==r[1:] and len(l[0])==len(r[0])
for x,y in zip(l[0],r[0]):assert {k:v for k,v in x.items() if k!='action'}=={k:v for k,v in y.items() if k!='action'}
scene={k:{**v,'selected_instructions':sum(len(x['episodes']) for x in new_rows if str(x['scene_id'])==k)} for k,v in V['by_scene'].items()}
report={**V,'created_at':time.time(),'status':'target_valid_view_built_actual_loader_and_sampler_pending','dataset_version':C.name,
 'usable_instructions':len(ids),'sample_counts':counts,'annotation_sha256':digests,'action_text_counts':classes,'all_samples_endpoint_STOP_count':len(ids),'by_scene':scene,
 'excluded_from_v2_instructions':V['excluded_from_v2_instructions']+len(bad),'parent_dataset':P.name,'parent_validation_sha256':sha(P/'validation_report.json'),
 'excluded_parent_instruction_count':len(bad),'excluded_parent_instructions':sorted(bad),'physical_route_set_unchanged':True,
 'all_selected_RGB_JPEG_hashes_verified_again':False,'RGB_full_JPEG_verification_inherited_from_parent':True,'all_selected_RGB_manifests_verified_again':True,
 'RGB_integrity_scope':'Parent C3 full253714JPEG hashes verified; identical physical route/frame/RGB paths retained, all manifests rechecked. No JPEG rehash is claimed for this new view.',
 'REC_labels_byte_identical_to_parent':True,'target_retention_audit_sha256':sha(audit_path),'minimum_retained_STOP_context_margin_tokens':minimum_margin,
 'maximum_ASCII_response_byte_plus_framing_bound':response_byte_bound,'actual_released_loader_pending':True,'formal_training_ready':False,'formal_training_submitted':False,
 'source_builder_sha256':sha(Path(__file__)),'old_datasets_modified':False,'cohort_selection':V['cohort_selection']+' Additionally exclude instructions with dropped STOP targets under unchanged released4096token preprocessing; do not rewrite instruction text or action labels.'}
save(C/'validation_report.json',report);save(C/'FORMAL_TRAINING_PENDING_REVIEW.json',{'at':time.time(),'scope':'Require real released loader, per-seed sampler qualification and concrete new C4 training configs. All parent holds remain.'})
save(R/'report/target_valid_cohort_v4.json',report)
print(json.dumps({'dataset':str(C),'routes':len(new_rows),'instructions':len(ids),'excluded_instructions':len(bad),'RGB_frames':frames,'sample_counts':counts,'minimum_context_margin':minimum_margin}),flush=True)
