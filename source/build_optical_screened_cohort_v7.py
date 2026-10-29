"""Build new immutable matched data only after every optical screen completes.

Original annotations/RGB and v2 labels remain immutable. The optical screen is
an explicitly limited geometry proxy, not physical collision certification.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,os,shutil,time
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root).resolve();V2=R/'dataset/complete-proxy-pass-metric-v2';C=R/'dataset/complete-optical-screened-metric-v3';A=R/'eval/optical_centerline_screen_all_selected_005'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False));q.replace(p)
plan=json.loads((A/'plan.json').read_text());done=json.loads((A/'exit.json').read_text());assert done['success'] and done['totals']['completed_routes']==2899 and done['totals']['completed_instructions']==7616
rows=json.loads((V2/'route_inventory.json').read_text());assert sha(V2/'route_inventory.json')==plan['route_inventory_sha256'];assert len(rows)==2899
assert not C.exists();C.mkdir();write(C/'BUILDING.json',{'time':time.time(),'source_sha256':sha(Path(__file__)),'screen_final_receipt_sha256':sha(A/'exit.json')})
selected=[];excluded=[];audit={}
for scene,record in done['completed_scenes'].items():
 e=Path(record['attempt']);assert e.resolve().is_relative_to((R/'eval').resolve());proto=json.loads((e/'protocol.json').read_text())
 assert proto['cohort_inventory_sha256']==plan['route_inventory_sha256'] and proto['probe_source_sha256']==plan['probe_sha256'] and proto['source_sha256'] in plan['allowed_worker_source_sha256']
 for p in sorted((e/'routes').glob('*.json')):
  x=json.loads(p.read_text());assert x['route_id'] not in audit;audit[x['route_id']]=(x,sha(p),str(p))
assert set(audit)=={x['route_id'] for x in rows}
for row in rows:
 d,h,p=audit[row['route_id']];assert len(d['episodes'])==len(row['episodes']) and set(d['episodes'])==set(row['episodes']);assert d['frames']==row['frame_count'] and d['GT_action_vs_reference_max_error_m']<1e-7
 if d['confirmed_optical_conflict_segments'] or d['unresolved_segments']:
  excluded.append({'route_id':row['route_id'],'scene_id':row['scene_id'],'episodes':row['episodes'],'confirmed_optical_conflict_segments':d['confirmed_optical_conflict_segments'],'unresolved_segments':d['unresolved_segments'],'screen_record_sha256':h,'screen_record':p})
 else:selected.append({**row,'optical_screen_record_sha256':h,'optical_screen_record':p})
assert selected and excluded and len(selected)+len(excluded)==2899
write(C/'route_inventory.json',selected);write(C/'excluded_route_inventory.json',excluded)
# Recheck the actual selected JPEG bytes against the immutable per-frame manifest.
def verify_rgb(row):
 folder=Path(row['rgb']);manifest=folder/'manifest.json';assert sha(manifest)==row['RGB_manifest_sha256'];d=json.loads(manifest.read_text());assert d['frame_count']==row['frame_count']==len(d['frames'])
 for i,x in enumerate(d['frames']):assert x['index']==i and sha(folder/f'frame_{i:03d}.jpg')==x['jpeg_sha256'],(row['route_id'],i)
 return row['frame_count']
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:verified_frames=sum(pool.map(verify_rgb,selected))
ids={eid for row in selected for eid in row['episodes']};assert len(ids)==sum(len(x['episodes']) for x in selected)
old=json.loads((V2/'validation_report.json').read_text());assert sha(V2/'validation_report.json')==json.loads((V2/'READY.json').read_text())['validation_report_sha256'];counts={};digests={};paired={};classes={}
for group in old['sample_counts']:
 src=V2/group;dst=C/group;sa=src/'data/aerialvln-s';da=dst/'data/aerialvln-s'
 for fn,h in old['annotation_sha256'][group].items():assert sha(sa/fn)==h, ('Frozen parent annotation changed',group,fn)
 samples=[x for x in json.loads((sa/'train_merged_triple.json').read_text()) if x['episode'] in ids]
 idx={k:v for k,v in json.loads((sa/'train_episodes2idx_merge.json').read_text()).items() if k in ids};instructions={k:v for k,v in json.loads((sa/'train_episode2instruction.json').read_text()).items() if k in ids};assert set(idx)==set(instructions)==ids
 stops=collections.Counter(x['episode'] for x in samples if x['action']=='The next action is stop.');assert set(stops)==ids and set(stops.values())=={1}
 assert all(x['episode'] in ids for x in samples)
 for name,obj in [('train_merged_triple.json',samples),('train_episodes2idx_merge.json',idx),('train_episode2instruction.json',instructions)]:write(da/name,obj)
 shutil.copy2(sa/'train_action_prob_weight.json',da/'train_action_prob_weight.json')
 rgb=dst/'Raw_data/aerialvln-s';rgb.parent.mkdir(parents=True);rgb.symlink_to(os.path.relpath((src/'Raw_data/aerialvln-s').resolve(),rgb.parent),target_is_directory=True)
 for eid in ids:assert (rgb/eid/'rgb').is_dir()
 counts[group]=len(samples);classes[group]=dict(collections.Counter(x['action'] for x in samples));digests[group]={p.name:sha(p) for p in da.glob('*.json')}
 if group.endswith('joint-boundary'):paired[group]=(samples,idx,instructions)
left=paired['gt-joint-boundary'];right=paired['rec-joint-boundary'];assert left[1:]==right[1:] and len(left[0])==len(right[0])
for l,r in zip(left[0],right[0]):assert {k:v for k,v in l.items() if k!='action'}=={k:v for k,v in r.items() if k!='action'}
old_by={str(e['episode_id']):e for e in json.loads((Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/inventory/official_train.json')).read_text())['episodes']};correct=total=exact=0;confusion=[[0]*7 for _ in range(7)]
for row in selected:
 p=V2/'route_labels'/(row['route_id']+'.json');label=json.loads(p.read_text());assert label['GT_poses_used_for_REC_labels'] is False and label['GT_actions_used_for_REC_labels'] is False
 dest=C/'route_labels'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest);assert sha(p)==sha(dest)
 gt=old_by[row['episodes'][0]]['actions'];rec=label['GT_free_REC_actions'];assert len(gt)==len(rec)==row['frame_count'] and rec[-1]==gt[-1]==0;total+=len(gt)-1;correct+=sum(x==y for x,y in zip(gt[:-1],rec[:-1]));exact+=int(gt==rec)
 for x,y in zip(gt[:-1],rec[:-1]):assert 1<=x<=7 and 1<=y<=7;confusion[x-1][y-1]+=1
class_scores=[]
for i in range(7):
 tp=confusion[i][i];fp=sum(confusion[j][i] for j in range(7) if j!=i);fn=sum(confusion[i][j] for j in range(7) if j!=i);class_scores.append({'action':i+1,'GT_support':sum(confusion[i]),'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.})
by_scene={}
for scene in sorted({x['scene_id'] for x in rows}):
 source=[x for x in rows if x['scene_id']==scene];keep=[x for x in selected if x['scene_id']==scene];drop=[x for x in excluded if x['scene_id']==scene]
 by_scene[str(scene)]={'v2_routes':len(source),'selected_routes':len(keep),'selected_instructions':sum(len(x['episodes']) for x in keep),'excluded_routes':len(drop),'route_retention_from_v2':len(keep)/len(source)}
report={**old,'created_at':time.time(),'status':'built_actual_JPEG_hashes_checked_official_loader_pending','dataset_version':C.name,'usable_routes':len(selected),'usable_instructions':len(ids),'physical_RGB_frames':verified_frames,'nonstop_actions':total,'correct_nonstop_actions':correct,'nonstop_agreement':correct/total,'confusion_GT_rows_REC_cols_1to7':confusion,'class_F1':class_scores,'macro_F1':sum(x['F1'] for x in class_scores)/7,'exact_sequence_routes':exact,'exact_sequence_route_fraction':exact/len(selected),'sample_counts':counts,'all_samples_endpoint_STOP_count':len(ids),'annotation_sha256':digests,'cohort_selection':'Frozen v2complete/proxy-pass paired routes with no confirmed body-centered optical surface conflict and no unresolved optical segment; not full official GT reproduction.','screen_scope':'Rendered central2x2surface proxy along GT translations; not a swept-body, rotation or collision-free certificate.','physical_legality_certified':False,'GT_and_simulator_used_for_research_cohort_selection':True,'GT_poses_used_for_REC_labels':False,'GT_actions_used_for_REC_labels':False,'REC_labels_byte_identical_to_parent':True,'all_selected_RGB_JPEG_hashes_verified_again':True,'old_datasets_modified':False,'source_candidate_inventory_sha256':sha(V2/'route_inventory.json'),'screen_final_receipt_sha256':sha(A/'exit.json'),'source_builder_sha256':sha(Path(__file__)),'excluded_from_v2_routes':len(excluded),'excluded_from_v2_instructions':sum(len(x['episodes']) for x in excluded),'by_scene':by_scene,'action_text_counts':classes,'formal_training_submitted':False,'formal_training_ready':False}
write(C/'validation_report.json',report);write(C/'FORMAL_TRAINING_PENDING_REVIEW.json',{'scope':'Structural dataset build only; require actual loader, sampler/budget and new reviewed training configs. Parentv2training hold is not cleared.','time':time.time(),'formal_submission_authorized_by_this_file':False});write(R/'report/optical_screened_cohort_v3.json',report)
print(json.dumps({'dataset':str(C),'routes':len(selected),'instructions':len(ids),'samples':counts,'RGB_verified_frames':verified_frames},indent=2),flush=True)
