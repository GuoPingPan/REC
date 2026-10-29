"""Explicit configured DLC run. Preparation does not submit any cloud job."""
from pathlib import Path
import argparse, hashlib, json, math, os, shutil, socket, struct
import subprocess, sys, tarfile, time, traceback

ap=argparse.ArgumentParser()
ap.add_argument('--config', required=True, type=Path)
a=ap.parse_args()
cfg=json.loads(a.config.read_text())
R=Path(cfg['root']).resolve()
E=Path(cfg['run_root']).resolve()
SOURCE=Path(cfg['frozen_source']).resolve()
nodes=cfg['nodes']
node=int(os.environ.get('NODE_RANK', os.environ.get('RANK','-1')))
assert 0<=node<nodes and os.environ.get('MASTER_ADDR') and os.environ.get('MASTER_PORT')
assert nodes==1 or os.environ['MASTER_ADDR'] not in ['localhost','127.0.0.1']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.%d.partial'%os.getpid())
    q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n'); q.replace(p)

try:
    assert cfg['status']=='reviewable_candidate_not_submitted'
    gate=json.loads((R/'eval/eight_action_stable_rotary_001/learning_gate.json').read_text())
    assert gate['passed'] and gate['summary']['exact_actions_correct']==8
    proof=R/'inventory/strict_release_collator_source.json'
    assert sha(proof)==cfg['frozen_source_manifest_sha256']
    manifest=json.loads(proof.read_text())
    assert Path(cfg['frozen_source']).name==Path(manifest['source']).name
    helper=R/'source/rotary_compat.py'
    entry=R/'source/formal_training_entry.py'
    assert sha(helper)==cfg['rotary_helper_sha256']
    assert sha(entry)==cfg['entry_sha256']
    assert sha(Path(__file__))==cfg['bootstrap_sha256']
    D=Path(cfg['dataset']).resolve()
    cohort=D.parent
    ready=json.loads((cohort/'READY.json').read_text())
    assert ready['status']=='ready'
    assert cohort.name=='complete-optical-target-valid-v4'
    assert not (cohort/'TRAINING_USE_HOLD.json').exists(), 'Dataset is on hold'
    post=R/'pipeline/optical_screened_dataset_v3_005/verified_dataset.json'
    assert sha(post)==cfg['postscreen_verification_sha256']
    verification=json.loads(post.read_text())
    assert verification['status']=='research_dataset_structurally_verified_new_training_configs_pending'
    final_screen=R/'eval/optical_centerline_screen_all_selected_005/exit.json'
    assert sha(final_screen)==cfg['screen_receipt_sha256']
    screen=json.loads(final_screen.read_text())
    assert screen['success'] and screen['totals']['completed_routes']==2899 and screen['totals']['completed_instructions']==7616
    sampler_helper=R/'source/sampler_seed_contract_v2.py'
    assert sha(sampler_helper)==cfg['sampler_contract_sha256']
    import importlib.util
    spec=importlib.util.spec_from_file_location('qualified_sampler_seed',sampler_helper)
    seed_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(seed_contract)
    sampler_proof=seed_contract.verify_sampler_seed(R,cfg)
    assert sha(cohort/'validation_report.json')==ready['validation_report_sha256']==cfg['dataset_validation_sha256']
    validation=json.loads((cohort/'validation_report.json').read_text())
    assert validation['sample_counts'][cfg['group']]==cfg['samples']
    for name,digest in validation['annotation_sha256'][cfg['group']].items():
        assert sha(D/'data/aerialvln-s'/name)==digest
    target_contract=R/'source/target_valid_data_contract_v1.py'
    assert sha(target_contract)==cfg['target_valid_contract_sha256']
    spec=importlib.util.spec_from_file_location('qualified_target_valid_data',target_contract)
    td=importlib.util.module_from_spec(spec);spec.loader.exec_module(td)
    target_proof=R/'report/target_valid_dataset_verification_001.json'
    assert sha(target_proof)==cfg['target_valid_verification_sha256']
    assert td.verify_target_valid_cohort(R)==json.loads(target_proof.read_text())['verification']
    M=Path(cfg['model']).resolve()
    assert json.loads((M/'READY.json').read_text())['all_files_size_and_hub_digest_verified']
    arc=Path(cfg['runtime_archive']).resolve()
    assert sha(arc)==cfg['runtime_archive_sha256']
    prefix=Path('/tmp/aeroact_official_runtime_0417_py310_v1')
    marker=prefix/'AEROACT_ARCHIVE_SHA256'
    if not marker.exists():
        assert not prefix.exists()
        with tarfile.open(arc) as t:
            for m in t.getmembers():
                assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                if m.issym() or m.islnk():
                    assert not Path(m.linkname).is_absolute()
                    base=Path('/tmp')/Path(m.name).parts[0]
                    target=(Path('/tmp')/m.linkname if m.islnk() else (Path('/tmp')/m.name).parent/m.linkname)
                    assert os.path.commonpath([str(base),os.path.normpath(str(target))])==str(base)
            t.extractall('/tmp')
        marker.write_text(cfg['runtime_archive_sha256'])
    assert marker.read_text()==cfg['runtime_archive_sha256']
    # Copy only individually recorded source files, never mutable pycache or old DLC launchers.
    W=Path('/tmp')/('aeroact_paper_%s_%s_seed%d_%s_node%d'%(hashlib.sha256(str(E).encode()).hexdigest()[:10],cfg['group'],cfg['seed'],cfg['attempt'],node))
    W.mkdir(exist_ok=False)
    for rel,digest in manifest['files'].items():
        src=SOURCE/rel
        assert sha(src)==digest, rel
        dst=W/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        assert sha(dst)==digest
    for rel,target in manifest['symlinks'].items():
        assert not Path(target).is_absolute()
        dst=W/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(target)
    data_root=W/'Dataset/AerialVLN-Dataset'
    assert data_root.is_dir() and not data_root.is_symlink()
    shutil.rmtree(data_root)
    data_root.symlink_to(D,target_is_directory=True)
    audit=W/'paper_audit';audit.mkdir()
    for src in [helper,entry]:
        shutil.copyfile(src,audit/src.name)
    protocol={k:cfg[k] for k in ['world_size','samples','optimizer_steps','per_GPU_batch','gradient_accumulation','seed','learning_rate','scope','rotary_mode']}
    # Prepare once before launch; all workers agree on this immutable protocol.
    if node==0:
        E.mkdir(parents=True,exist_ok=True)
        assert not (E/'protocol.json').exists(), 'Existing run: inspect, never duplicate.'
        write(E/'protocol.json',protocol)
    deadline=time.monotonic()+600
    while not (E/'protocol.json').exists():
        assert time.monotonic()<deadline
        time.sleep(1)
    assert json.loads((E/'protocol.json').read_text())==protocol
    (E/'logs').mkdir(exist_ok=True)
    (E/'debug').mkdir(exist_ok=True)
    env=dict(os.environ)
    env.update(PAPER_TRAIN_ROOT=str(E),PAPER_FROZEN_SOURCE=str(W),PYTHONPATH=str(audit)+os.pathsep+str(W),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',WANDB_MODE='disabled',OMP_NUM_THREADS='8',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1',TORCH_EXTENSIONS_DIR='/tmp/aeroact_paper_extensions_node%d'%node,
        PATH=str(prefix/'bin')+os.pathsep+env.get('PATH',''))
    hw="import torch,json,socket; d={'host':socket.gethostname(),'torch':torch.__version__,'gpu_count':torch.cuda.device_count(),'GPUs':[{'name':torch.cuda.get_device_name(i),'memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]};assert d['gpu_count']==8;print(json.dumps(d))"
    rr=subprocess.run([str(prefix/'bin/python'),'-c',hw],env=env,capture_output=True,text=True,check=True)
    hardware=json.loads(rr.stdout.strip().splitlines()[-1])
    write(E/'debug'/('hardware-node-%d.json'%node),hardware)
    args=list(cfg['training_args'])
    cmd=[str(prefix/'bin/python'),'-m','torch.distributed.run','--nnodes='+str(nodes),
        '--nproc_per_node=8','--node_rank='+str(node),'--master_addr='+env['MASTER_ADDR'],
        '--master_port='+env['MASTER_PORT'],str(audit/'formal_training_entry.py')]+args
    write(E/'debug'/('launch-node-%d.json'%node),{'node':node,'host':socket.gethostname(),
        'command':cmd,'config_sha256':sha(a.config),'dataset_validation_sha256':cfg['dataset_validation_sha256'],
        'source_manifest_sha256':cfg['frozen_source_manifest_sha256'],'rotary_mode':cfg['rotary_mode'],'time':time.time()})
    with (E/'logs'/('node-%d.log'%node)).open('w') as log:
        child=subprocess.Popen(cmd,cwd=W,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in child.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=child.wait()
    write(E/'debug'/('node-exit-%d.json'%node),{'exit_code':code,'time':time.time()})
    assert code==0
    if node==0:
        deadline=time.monotonic()+900
        while not all((E/'debug'/('node-exit-%d.json'%n)).exists() for n in range(nodes)):
            assert time.monotonic()<deadline;time.sleep(2)
        for n in range(nodes):
            assert json.loads((E/'debug'/('node-exit-%d.json'%n)).read_text())['exit_code']==0
        for rank in range(cfg['world_size']):
            st=json.loads((E/'debug'/('trainer-ended-rank-%d.json'%rank)).read_text())
            assert st['steps']==cfg['optimizer_steps']
            assert (E/'debug'/('process-complete-rank-%d.json'%rank)).exists()
        steps=[json.loads(line) for line in (E/'logs/optimizer-steps.jsonl').read_text().splitlines()]
        assert [s['step'] for s in steps]==list(range(1,cfg['optimizer_steps']+1))
        C=Path(cfg['output_dir'])
        elements=0;headers={}
        for p in sorted(C.rglob('*.safetensors')):
            # Exclude intermediate checkpoint folders when counting the final model.
            if any(part.startswith('checkpoint-') for part in p.relative_to(C).parts):continue
            with p.open('rb') as f:
                hs=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(hs))
            tensors={k:v for k,v in h.items() if k!='__metadata__'}
            assert p.stat().st_size==8+hs+max(v['data_offsets'][1] for v in tensors.values())
            elements+=sum(math.prod(v['shape']) for v in tensors.values())
            headers[p]=tensors
        assert elements==8102301504 and len(headers)==6
        for index in C.rglob('*.safetensors.index.json'):
            if any(part.startswith('checkpoint-') for part in index.relative_to(C).parts):continue
            for name,shard in json.loads(index.read_text())['weight_map'].items():
                assert name in headers[index.parent/shard]
        write(E/'debug/training-process-complete.json',{'all_nodes_exit_zero':True,
            'ranks_completed':cfg['world_size'],'optimizer_steps':cfg['optimizer_steps'],
            'final_checkpoint':str(C),'tensor_elements':elements,'weight_files':len(headers),
            'rotary_mode':cfg['rotary_mode'],'time':time.time(),
            'platform_success_and_actual_reload_still_required':True})
except BaseException:
    write(E/'debug'/('bootstrap-error-node-%d.json'%node),{'traceback':traceback.format_exc(),'time':time.time()})
    raise
