"""Real GPU feature/preparation audit. No training or navigation evaluation."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,sys,time,types
from unittest import mock
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
a=ap.parse_args();R=Path(a.root).resolve();C=R/'dataset/complete-optical-screened-metric-v3'
S=Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source'])
OUT=R/'eval/multimodal_target_retention_002';OUT.mkdir(exist_ok=False);(OUT/'runtime').mkdir()
lock=open('/tmp/aerialvln_paper_shape_probe_gpu0.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0')
sys.path[:0]=[str(R/'source'),str(S)];sys.argv=[sys.argv[0]]
import torch
from transformers import set_seed
from llava import conversation as conv
import llava.data.dataset as md
from llava.train.args import DataArguments
from llava.train.transformer_normalize_monkey_patch import patched_normalize
from policy_eval_inference_entry_v1 import single_process_inference
torch.set_num_threads(4);set_seed(10);conv.default_conversation=conv.conv_templates['auto']
M=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/runtime/models/NVILA-Lite-8B-ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d')
spec=importlib.util.spec_from_file_location('frozen_shape_eval',S/'eval.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
text=json.loads((R/'report/instruction_target_truncation_v1.json').read_text())
assert all(v['supervised_tokens_before']==v['last_supervised_token_index']-v['first_supervised_token_index']+1 for v in text['records']), 'Projection requires contiguous supervised spans'
ranked=sorted(text['records'],key=lambda x:x['input_tokens_before_collator'],reverse=True)
# Cover every large-text case and the longest remaining ordinary navigation text.
eids=[x['episode'] for x in ranked[:9]]
D=C/'gt-joint-boundary';A=D/'data/aerialvln-s'
rows=json.loads((A/'train_merged_triple.json').read_text());stop_by={x['episode']:x for x in rows if x['action']=='The next action is stop.'}
with single_process_inference(OUT/'runtime'):
    tok,model,processor,_=ev.load_pretrained_model(str(M),'final_model',None,device='cuda',device_map=None)
    tok.model_max_length=4096;tok.padding_side='right'
    model.eval();model.to(device='cuda',dtype=torch.float16)
    model.llm.config.tokenizer_model_max_length=4096
    da=DataArguments(image_aspect_ratio='dynamic',num_video_frames=8,idx_type='merge',hisotry_queue_len=0,temporal_aggregation_frames=8,label_balance=True,label_balance_type='v1',is_multimodal=True)
    da.image_processor=processor
    ds=md.LazyUpdateTripleMergeAirVLNDataset.__new__(md.LazyUpdateTripleMergeAirVLNDataset)
    ds.list_data_dict=[stop_by[e] for e in eids];ds.ids_merge=json.loads((A/'train_episodes2idx_merge.json').read_text())
    ds.episode2instruction=json.loads((A/'train_episode2instruction.json').read_text())
    ds.action_prob_weight=json.loads((A/'train_action_prob_weight.json').read_text())
    ds.tokenizer=tok;ds.data_args=da;ds.image_folder=str(D/'Raw_data/aerialvln-s')
    ds.stop_prob=0.;ds.idx_type='merge';ds.label_balance=True;ds.label_balance_type='v1'
    ds.hisotry_queue_len=0;ds.temporal_aggregation_frames=8
    original_load=md.LazyUpdateTripleMergeAirVLNDataset._load_video
    def checked_load(*args,**kwargs):
        images,valid=original_load(*args,**kwargs);assert valid;return images,valid
    md.LazyUpdateTripleMergeAirVLNDataset._load_video=staticmethod(checked_load)
    collator=md.DataCollatorForSupervisedDataset(tokenizer=tok,data_args=da)
    results=[]
    with torch.inference_mode():
        for i,eid in enumerate(eids):
            with mock.patch('transformers.image_processing_utils.normalize',new=patched_normalize):x=ds[i]
            assert tuple(x['image'].shape)==(8,3,448,448)
            batch=collator([x]);ids=batch['input_ids'].cuda();labels=batch['labels'].cuda();mask=batch['attention_mask'].cuda()
            images=[im.to(device='cuda',dtype=torch.float16) for im in batch['images']]
            features=model.encode_images(torch.cat(images,dim=0))
            original_encode=model.encode_images
            model.encode_images=types.MethodType(lambda self,ims:features,model)
            try:
                model.llm.config.tokenizer_model_max_length=None
                uncut=model.prepare_inputs_labels_for_multimodal(ids,None,mask,None,labels,images)
                uncut_shape=list(uncut[4].shape);loc=torch.nonzero(uncut[5][0]!=-100).flatten()
                positions=loc.cpu().tolist();del uncut
                model.llm.config.tokenizer_model_max_length=4096
                cut=model.prepare_inputs_labels_for_multimodal(ids,None,mask,None,labels,images)
                item={'episode':eid,'real_RGB_decode_succeeded':True,'image_tensor_shape':list(x['image'].shape),
                      'feature_shape':list(features.shape),'raw_text_tokens':len(x['input_ids']),
                      'post_collator_tokens':int(mask.sum()),
                      'supervised_before_collator':int((x['labels']!=-100).sum()),
                      'supervised_after_collator':int((labels!=-100).sum()),
                      'expanded_uncut_shape':uncut_shape,'uncut_supervised_positions':positions,
                      'expanded_capped_shape':list(cut[4].shape),
                      'supervised_after_embedding_cap':int((cut[5]!=-100).sum())}
                results.append(item);(OUT/'cases.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(item),flush=True);del cut
            finally:model.encode_images=original_encode
            del features,ids,labels,mask,images,x,batch
    surplus={v['expanded_uncut_shape'][1]-v['post_collator_tokens'] for v in results}
    assert len(surplus)==1
    extra=surplus.pop()
    projected=[{**v,'supervised_after_embedding_cap_predicted':sum(p+extra<4096 for p in range(v['first_supervised_token_index'],v['last_supervised_token_index']+1))} for v in text['records']]
    observed={v['episode']:v['supervised_after_embedding_cap'] for v in results}
    assert all(observed[v['episode']]==v['supervised_after_embedding_cap_predicted'] for v in projected if v['episode'] in observed)
    affected=[v for v in projected if v['supervised_after_embedding_cap_predicted']<v['supervised_tokens_before']]
    report={'time':time.time(),'dataset':C.name,'dataset_validation_sha256':text['dataset_validation_sha256'],
            'scope':'Real base-model FP16 visual encoding and released multimodal preparation on9selected real RGB cases; no LLM forward, training or policy navigation.',
            'GPU_name':torch.cuda.get_device_name(0),'feature_surplus_per_fixed8frame_input':extra,
            'model_max_length':4096,'cases':results,'affected_STOP_instructions':affected,
            'all_instruction_projection_scope':'Projection of audited token positions using constant feature surplus;9actual embedding cases checked, not5920GPUforwards.',
            'data_or_source_modified':False,'new_paid_jobs':0,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'llava_arch_sha256':hashlib.sha256((S/'llava/model/llava_arch.py').read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'summary':str(OUT/'summary.json'),'affected_STOP_instructions':len(affected),'feature_surplus':extra}),flush=True)
