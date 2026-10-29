"""Read-only qualification of the target-valid C4 paired training view."""
from pathlib import Path
import argparse,hashlib,json
DATASET='complete-optical-target-valid-v4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_target_valid_cohort(root):
 R=Path(root).resolve();C=R/'dataset'/DATASET;P=R/'dataset/complete-optical-screened-metric-v3'
 for rel in ['eval/optical_centerline_screen_all_selected_005','pipeline/optical_screened_dataset_v3_005','pipeline/verified_data_analysis_002']:
  assert read(R/rel/'exit.json')['success'] and not (R/rel/'failure.json').exists()
 assert read(R/'report/target_valid_cohort_v4_build_exit.json')['returncode']==0
 assert read(R/'pipeline/target_valid_runtime_checks_001/exit.json')['success']
 V=read(C/'validation_report.json');ready=read(C/'READY.json');pv=read(P/'validation_report.json')
 assert ready['status']=='ready' and ready['validation_report_sha256']==sha(C/'validation_report.json')
 assert read(P/'READY.json')['validation_report_sha256']==sha(P/'validation_report.json')==V['parent_validation_sha256']
 assert (P/'TRAINING_USE_HOLD.json').exists() and not (C/'TRAINING_USE_HOLD.json').exists()
 assert V['physical_legality_certified'] is False
 assert pv['all_selected_RGB_JPEG_hashes_verified_again'] and V['RGB_full_JPEG_verification_inherited_from_parent'] and V['all_selected_RGB_manifests_verified_again']
 assert V['REC_labels_byte_identical_to_parent'] and V['physical_route_set_unchanged']
 assert (C/'route_labels').resolve()==(P/'route_labels').resolve()
 source=read(R/'inventory/strict_release_collator_source.json')
 loader_path=R/'report'/('official_loader_preflight_'+DATASET+'.json');loader=read(loader_path)
 assert sha(loader_path)==ready['official_loader_preflight_sha256'] and loader['passed'] and loader['joint_boundary_pair_same_inputs_confirmed']
 assert loader['source_dataset_py_sha256']==source['files']['llava/data/dataset.py']
 tpath=R/'report'/('instruction_target_truncation_'+DATASET+'_v2.json');text=read(tpath)
 assert text['dataset_validation_sha256']==sha(C/'validation_report.json')
 assert text['checked_STOP_episodes']==V['usable_instructions'] and text['STOP_targets_fully_discarded']==text['STOP_targets_partially_discarded']==0
 gp=R/'eval/multimodal_target_retention_002/summary.json';gpu=read(gp)
 assert read(R/'report/multimodal_target_retention_002_exit.json')['returncode']==0
 assert sha(gp)==V['target_retention_audit_sha256'] and gpu['dataset_validation_sha256']==sha(P/'validation_report.json')
 assert len(gpu['cases'])==9 and gpu['model_max_length']==4096
 assert all(x['real_RGB_decode_succeeded'] and x['image_tensor_shape']==[8,3,448,448] for x in gpu['cases'])
 bad={x['episode'] for x in gpu['affected_STOP_instructions']}
 assert bad==set(V['excluded_parent_instructions']) and len(bad)==V['excluded_parent_instruction_count']==5
 rows=read(C/'route_inventory.json');parent={x['route_id']:x for x in read(P/'route_inventory.json')}
 assert len(rows)==len(parent)==V['usable_routes']==2264 and {x['route_id'] for x in rows}==set(parent)
 ids=set()
 for x in rows:
  y=parent[x['route_id']];assert x['episodes']==[e for e in y['episodes'] if e not in bad]
  for key in ['frame_count','rgb','RGB_manifest_sha256','optical_screen_record_sha256']:assert x[key]==y[key]
  assert not ids.intersection(x['episodes']);ids.update(x['episodes'])
 assert len(ids)==V['usable_instructions']==5915 and sum(x['frame_count'] for x in rows)==V['physical_RGB_frames']==253714
 surplus=gpu['feature_surplus_per_fixed8frame_input']
 margins=[4095-x['last_supervised_token_index']-surplus for x in text['records']]
 assert len(margins)==len(ids) and {x['episode'] for x in text['records']}==ids
 assert min(margins)==V['minimum_retained_STOP_context_margin_tokens']>V['maximum_ASCII_response_byte_plus_framing_bound']
 for group,count in V['sample_counts'].items():
  assert loader['groups'][group]['loaded_samples']==count and loader['groups'][group]['episodes']==len(ids)
  for n,digest in V['annotation_sha256'][group].items():assert sha(C/group/'data/aerialvln-s'/n)==digest
  assert (C/group/'Raw_data/aerialvln-s').resolve()==(P/group/'Raw_data/aerialvln-s').resolve()
 assert V['sample_counts']['gt-joint-boundary']==V['sample_counts']['rec-joint-boundary']==351983
 return {'dataset':DATASET,'routes':len(rows),'instructions':len(ids),'RGB_frames':V['physical_RGB_frames'],
  'samples':V['sample_counts'],'validation_report_sha256':sha(C/'validation_report.json'),
  'route_inventory_sha256':sha(C/'route_inventory.json'),'parent_validation_sha256':sha(P/'validation_report.json'),
  'loader_sha256':sha(loader_path),'all_STOP_text_audit_sha256':sha(tpath),'GPU_embedding_audit_sha256':sha(gp),
  'minimum_context_margin_tokens':min(margins),'excluded_instructions':sorted(bad),
  'physical_collision_free_certified':False,'known_primitive_clock_and_clip_endpoint_priors':True,
  'contract_source_sha256':sha(Path(__file__))}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root).resolve()
 value=verify_target_valid_cohort(R);out=R/'report/target_valid_dataset_verification_001.json'
 assert not out.exists()
 with out.open('x') as f:json.dump({'status':'target_valid_paired_dataset_verified','verification':value},f,indent=2)
 print(json.dumps(value),flush=True)
