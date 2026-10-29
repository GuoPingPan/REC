"""Independently expand every actual GT target against freshly anchored official data.

Reads only existing data/source/checkpoints; creates a new audit directory.
"""
from pathlib import Path
import argparse,ast,collections,hashlib,json,tarfile,time,zipfile

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();R=Path(a.root)
U=R.parent.parent.parent.parent
# Resolve from the fixed experiment layout, never infer data from working directory.
U=Path('/mnt/data/users/kindel/uav')
O=U/'experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official'
N=U/'experiments/aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery'
C=R/'dataset/complete-optical-target-valid-v4';K=U/'datasets/aerialvln-s/kaggle-v3'
A=R/'analysis/official_GT_end_to_end_002';A.mkdir(parents=True,exist_ok=False)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fresh=read(R/'analysis/official_GT_external_recheck_001/receipt.json');assert fresh['all_succeeded']
up=O/'source/upstream-0417d39d4a739ed985ef33b9cb9be4327543640e'
bindings=[]
for x in fresh['sources']:
    p=K/x['path'] if x['path'].endswith('.zip') else up/x['path']
    assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
    bindings.append({'external_url':x['url'],'remote_file':str(p),'sha256':x['sha256'],'fresh_download_matches_cached_bytes':True})
print(json.dumps({'stage':'fresh_upstream_and_Kaggle_bytes_verified'}),flush=True)
original={};rawproof={}
with zipfile.ZipFile(K/'aerialvln-s-v3.zip') as z:
    for split in ['train','val_seen','val_unseen']:
        matches=[n for n in z.namelist() if Path(n).name==split+'.json'];assert len(matches)==1
        b=z.read(matches[0]);h=hashlib.sha256(b).hexdigest();inventory=O/'inventory'/('official_'+split+'.json')
        assert (K/(split+'.json')).read_bytes()==b and inventory.read_bytes()==b
        paths=[str(K/(split+'.json')),str(inventory)]
        if split!='train':
            q=O/'eval/official_annotations'/(split+'.json');assert q.read_bytes()==b;paths.append(str(q))
        original[split]=json.loads(b)['episodes'];rawproof[split]={'ZIP_member':matches[0],'sha256':h,'episodes':len(original[split]),'byte_identical_files':paths}
by={str(e['episode_id']):e for e in original['train']};assert len(by)==10113
pub=up/'Dataset/AerialVLN-Dataset/data/aerialvln-s'
public_instructions=read(pub/'train_episode2instruction.json');public_idx=read(pub/'train_episodes2idx_merge.json')
assert set(public_instructions)==set(by)
assert all(public_instructions[eid]==e['instruction']['instruction_text'] for eid,e in by.items())
verbs={1:('move forward',5,'units'),2:('turn left',15,'degrees'),3:('turn right',15,'degrees'),4:('ascend',2,'units'),5:('descend',2,'units'),6:('move left',5,'units'),7:('move right',5,'units')}
answer={'The next action is stop.':(0,1)}
for k,(v,amount,unit) in verbs.items():
    for n in [1,2,3]:answer[f'The next action is {v} {amount*n} {unit}.']=(k,n)
def native(ep):
    out=[];acts=ep['actions'];t=0
    while t<len(acts):
        k=acts[t];n=1
        while t+n<len(acts) and n<3 and acts[t+n]==k:n+=1
        txt='The next action is stop.' if k==0 else 'The next action is %s %s %s.'%(verbs[k][0],verbs[k][1]*n,verbs[k][2])
        out.append({'episode':str(ep['episode_id']),'timestep':t,'action':txt,'next_timestep':t if k==0 else t+n});t+=n
    return out
rows=read(C/'route_inventory.json');ids={str(eid) for row in rows for eid in row['episodes']}
assert len(rows)==2264 and len(ids)==5915 and ids<=set(by)
assert sum(row['frame_count']-1 for row in rows)==251450
route_proof=[]
for row in rows:
    es=[by[str(eid)] for eid in row['episodes']];gt=es[0]['actions'];rid=row['route_id']
    assert all(e['actions']==gt and e['reference_path']==es[0]['reference_path'] and e['start_position']==es[0]['start_position'] and e['start_rotation']==es[0]['start_rotation'] for e in es)
    aa=N/'routes'/rid/'attempt-001/actions.json';d=read(aa)
    assert d['GT_observed_transition_actions']==gt[:-1] and len(gt)==row['frame_count'] and gt[-1]==0
    manifest=Path(row['rgb'])/'manifest.json';assert sha(manifest)==row['RGB_manifest_sha256']
    route_proof.append({'route_id':rid,'official_actions_match_observed':True,'actions_sha256':sha(aa),'RGB_manifest_sha256':sha(manifest)})
print(json.dumps({'stage':'all_2264_route_GT_action_sources_verified'}),flush=True)
groups={};seqs={};manifest=read(C/'validation_report.json')
for group in ['gt-native-merge','gt-joint-boundary']:
    P=C/group/'data/aerialvln-s'
    for fn,h in manifest['annotation_sha256'][group].items():assert sha(P/fn)==h
    xs=read(P/'train_merged_triple.json');inst=read(P/'train_episode2instruction.json');idx=read(P/'train_episodes2idx_merge.json')
    assert set(inst)==set(idx)==ids and all(inst[eid]==by[eid]['instruction']['instruction_text'] for eid in ids)
    assert read(P/'train_action_prob_weight.json')==read(pub/'train_action_prob_weight.json')
    byid=collections.defaultdict(list)
    for x in xs:byid[str(x['episode'])].append(x)
    assert set(byid)==ids
    primitives=stops=0;span=collections.Counter()
    for eid,ss in byid.items():
        expanded=[];t=0;observed_idx=[]
        for x in ss:
            assert set(x)=={'episode','timestep','next_timestep','action'} and x['timestep']==t
            act,n=answer[x['action']];observed_idx.append(t);span[n]+=1
            if act==0:
                assert x is ss[-1] and x['next_timestep']==t and t==len(by[eid]['actions'])-1;stops+=1
            else:assert x['next_timestep']==t+n
            expanded.extend([act]*n);t+=n
        assert expanded==by[eid]['actions'],(group,eid)
        assert observed_idx==idx[eid]
        primitives+=len(expanded)
        if group=='gt-native-merge':assert ss==native(by[eid]),eid
    seqs[group]=byid
    groups[group]={'samples':len(xs),'instructions':len(byid),'expanded_actions_including_Stop':primitives,
                   'all_expanded_GT_actions_equal_official':True,'all_instructions_verbatim_official':True,
                   'Stop_count':stops,'repeat_counts':dict(span),'public_weight_file_JSON_values_equal':True,'public_weight_file_byte_identical':sha(P/'train_action_prob_weight.json')==sha(pub/'train_action_prob_weight.json'),
                   'annotation_sha256':{p.name:sha(p) for p in P.glob('*.json')}}
    print(json.dumps({'stage':group,'verified_samples':len(xs)}),flush=True)
# Compare original published keyframe indices with native reconstruction and the actually trained joint view.
index_comparison={};examples=[]
for group,byid in seqs.items():
    available=missing=equal=0;bad=[];changed_from_native=0;added=removed=0
    for eid,ss in byid.items():
        starts=[x['timestep'] for x in ss];ns=[x['timestep'] for x in seqs['gt-native-merge'][eid]]
        changed_from_native+=starts!=ns;added+=len(set(starts)-set(ns));removed+=len(set(ns)-set(starts))
        if eid not in public_idx:missing+=1
        else:
            available+=1;equal+=starts==public_idx[eid]
            if starts!=public_idx[eid] and len(bad)<3:bad.append(eid)
    index_comparison[group]={'retained_episode_indices_in_public_release':available,'retained_episode_indices_absent_from_public_release':missing,
        'exact_index_list_matches_to_public':equal,'index_lists_different_from_public':available-equal,'different_example_IDs':bad,
        'index_lists_different_from_native_GT':changed_from_native,'start_indices_added_vs_native':added,'start_indices_removed_vs_native':removed}
for eid,ss in seqs['gt-joint-boundary'].items():
    nn=seqs['gt-native-merge'][eid]
    if ss!=nn:
        examples.append({'episode_id':eid,'official_actions_sha256':hashlib.sha256(json.dumps(by[eid]['actions']).encode()).hexdigest(),
                         'official_actions':by[eid]['actions'],'native_GT_samples':nn,'actual_joint_GT_samples':ss})
        if len(examples)==3:break
# Earlier full navigation-only baseline is independently reconstructed, not an official released full sample file.
base=O/'dataset/official-s-full-64a06ef2-aeroact0417-exclude6-v1/data/aerialvln-s'
base_rows=read(base/'train_merged_triple.json');base_by=collections.defaultdict(list)
for x in base_rows:
    if str(x['episode']) in ids:base_by[str(x['episode'])].append(x)
assert set(base_by)==ids
assert all(base_by[eid]==seqs['gt-native-merge'][eid] for eid in ids)
demos=read(pub/'train_merged_triple.json');assert len(demos)==7
assert all(x in native(by[str(x['episode'])]) for x in demos)
training=[]
for seed in [10,11,12]:
    for group in ['gt-joint-boundary','rec-joint-boundary']:
        f=R/'training_candidates_v2/paired-seed-candidates-v4/4nodes'/group/f'seed-{seed}/attempt-001/training_config.json'
        cfg=read(f);done=read(f.parent/'debug/completion_verified.json')
        assert done['training_config_sha256']==sha(f) and done['platform_status']=='Succeeded'
        assert cfg['group']==group and cfg['seed']==seed and cfg['dataset'].endswith('/complete-optical-target-valid-v4/'+group)
        assert cfg['dataset_validation_sha256']==sha(C/'validation_report.json') and cfg['samples']==351983
        training.append({'seed':seed,'group':group,'dataset':cfg['dataset'],'samples':cfg['samples'],'config_sha256':sha(f),'completion_sha256':sha(f.parent/'debug/completion_verified.json'),'platform_status':done['platform_status']})
# Recheck archived upstream against the actual frozen source, with full file coverage.
Smanifest=read(R/'inventory/strict_release_collator_source.json');source=Path(Smanifest['source']);Umanifest=read(O/'inventory/upstream_source_manifest.json')
archive=O/'inventory/aeroact-0417d39d4a739ed985ef33b9cb9be4327543640e.tar.gz';assert sha(archive)==Umanifest['archive_sha256']
same=[];changed=[];extra=[]
with tarfile.open(archive) as tar:
    items={m.name.split('/',1)[1]:m for m in tar.getmembers() if m.isfile() and '/' in m.name}
    for rel,h in Smanifest['files'].items():
        b=(source/rel).read_bytes();assert hashlib.sha256(b).hexdigest()==h
        if rel not in items:extra.append(rel);continue
        old=tar.extractfile(items[rel]).read();assert hashlib.sha256(old).hexdigest()==Umanifest['files'][rel]
        if old==b:same.append(rel)
        else:changed.append({'file':rel,'upstream_sha256':hashlib.sha256(old).hexdigest(),'used_sha256':h,
                             'Python_AST_equal':ast.dump(ast.parse(old.decode('utf8',errors='replace')),include_attributes=False)==ast.dump(ast.parse(b.decode('utf8',errors='replace')),include_attributes=False) if rel.endswith('.py') else None})
    missing=sorted(set(items)-set(Smanifest['files']))
case_id='3GLB5JMZF6TB8PVZ3QXVVRVLPPUDG2';case=next(x for x in original['val_unseen'] if str(x['episode_id'])==case_id)
extracted=read(R/'analysis/paired_case_unseen_seed12_scene09_001/official_episode.json');assert extracted==case
result={'created_at':time.time(),'status':'verified','fresh_external_sources':bindings,'raw_official_file_checks':rawproof,
        'GT_route_sources':{'routes':len(rows),'physical_nonstop_actions':251450,'route_action_mismatch_count':0,'physical_RGB_manifests_verified':len(rows),'all_RGB_pixels_rehashed_in_this_audit':False},
        'actual_GT_views':groups,'published_index_comparison':index_comparison,
        'derived_full_baseline_filtered_to_C4_exactly_matches_native_GT':True,'public_action_demo_rows_verified':len(demos),
        'public_full_action_sample_file_available':False,
        'training_config_and_completion_bindings':training,
        'source_difference':{'pinned_commit':fresh['commit'],'upstream_files':len(items),'byte_identical_files':len(same),'changed_files':changed,'added_files':extra,'missing_files':missing},
        'selected_case_official_episode_payload_equal':True,'selected_case_episode_id':case_id,
        'important_distinctions':['No primitive GT action, instruction, terminal Stop, or official evaluation episode payload was changed in the retained data.',
           'The actual GT training view is a filtered, rerendered, jointly segmented research control. It is not a byte-for-byte official training dataset or an official published-model reproduction.',
           'Joint segmentation changes observation/history keyframes and action durations. Its effects versus native GT segmentation have not been measured by a completed native-merge training ablation.',
           'Fresh external sources anchor the original ZIP and selected upstream release files; remaining frozen code files are compared against the preserved pinned release archive.',
           'This audit does not validate semantic correctness of every original instruction or infer why REC improves mean SR.'],
        'all_data_and_code_inputs_read_only':True,'new_training_or_eval_jobs':0,
        'source_sha256':sha(Path(__file__))}
for name,value in [('summary.json',result),('route_source_proofs.json',route_proof),('segmentation_difference_examples.json',examples)]:
    (A/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(A/'exit.json').write_text(json.dumps({'success':True,'source_sha256':sha(Path(__file__)),
  'files_sha256':{p.name:sha(p) for p in A.iterdir() if p.is_file()}},indent=2)+'\n')
print(json.dumps({'output':str(A),'GT_views':groups,'published_index_comparison':index_comparison,'source_difference':result['source_difference']},ensure_ascii=False),flush=True)
