"""Read-only real text preprocessing/collator audit; synthetic image tensors only.

This does not qualify multimodal feature expansion or image decoding. Its purpose
is to locate text targets discarded at the released collator's 4096-token cap.
"""
from pathlib import Path
import argparse, collections, hashlib, json, os, random, sys, time

ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
ap.add_argument('--dataset',required=True)
a=ap.parse_args();R=Path(a.root).resolve()
C=R/'dataset'/a.dataset
validation=json.loads((C/'validation_report.json').read_text())
S=Path(json.loads((R/'inventory/strict_release_collator_source.json').read_text())['source'])
out=R/'report'/('instruction_target_truncation_'+a.dataset+'_v2.json')
assert not out.exists(), 'Preserve earlier audit'
assert json.loads((R/'pipeline/optical_screened_dataset_v3_005/exit.json').read_text())['success']
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(S))
import torch
from transformers import AutoTokenizer,set_seed
from llava import conversation as conv
import llava.data.dataset as module
from llava.train.args import DataArguments

torch.set_num_threads(2);set_seed(10)
M=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/runtime/models/NVILA-Lite-8B-ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d')
tok=AutoTokenizer.from_pretrained(str(M/'llm'),model_max_length=4096,padding_side='right',use_fast=False,legacy=False)
conv.default_conversation=conv.conv_templates['auto']
D=C/'gt-joint-boundary';A=D/'data/aerialvln-s'
rows=json.loads((A/'train_merged_triple.json').read_text())
stops=[x for x in rows if x['action']=='The next action is stop.']
instructions=json.loads((A/'train_episode2instruction.json').read_text())
assert len(stops)==len(instructions)==validation['usable_instructions']
assert {x['episode'] for x in stops}==set(instructions)
da=DataArguments(image_aspect_ratio='dynamic',num_video_frames=8,idx_type='merge',hisotry_queue_len=0,temporal_aggregation_frames=8,label_balance=True,label_balance_type='v1',is_multimodal=True)
ds=module.LazyUpdateTripleMergeAirVLNDataset.__new__(module.LazyUpdateTripleMergeAirVLNDataset)
ds.list_data_dict=stops;ds.ids_merge=json.loads((A/'train_episodes2idx_merge.json').read_text())
ds.episode2instruction=instructions;ds.action_prob_weight=json.loads((A/'train_action_prob_weight.json').read_text())
ds.tokenizer=tok;ds.data_args=da;ds.image_folder=str(D/'Raw_data/aerialvln-s')
ds.stop_prob=0.;ds.idx_type='merge';ds.label_balance=True;ds.label_balance_type='v1'
ds.hisotry_queue_len=0;ds.temporal_aggregation_frames=8
module.LazyUpdateTripleMergeAirVLNDataset._load_video=staticmethod(lambda paths,n,args:([None]*n,True))
module.process_image=lambda image,args,folder:torch.zeros((3,1,1))
collator=module.DataCollatorForSupervisedDataset(tokenizer=tok,data_args=da)
records=[]
for i,row in enumerate(stops):
    x=ds[i];batch=collator([x]);valid=torch.nonzero(x['labels']!=-100).flatten()
    n=int(valid.numel());m=int((batch['labels']!=-100).sum())
    assert n>0, ('Original text preprocessing masked target',row['episode'])
    records.append({'episode':row['episode'],'instruction_chars':len(instructions[row['episode']]),
                    'input_tokens_before_collator':len(x['input_ids']),
                    'first_supervised_token_index':int(valid[0]),
                    'last_supervised_token_index':int(valid[-1]),
                    'supervised_tokens_before':n,'supervised_tokens_after':m,
                    'all_target_tokens_discarded':m==0,'target_partially_discarded':0<m<n})
    if (i+1)%200==0:print(json.dumps({'checked':i+1,'total':len(stops)}),flush=True)
counts={}
for group in ['gt-native-merge','rec-native-merge','gt-joint-boundary','rec-joint-boundary']:
    xs=rows if group=='gt-joint-boundary' else json.loads((C/group/'data/aerialvln-s/train_merged_triple.json').read_text())
    by_episode=collections.Counter(x['episode'] for x in xs)
    bad={x['episode'] for x in records if x['all_target_tokens_discarded']}
    # All actions have the same user prompt, but only the STOP target was run.
    counts[group]={'all_samples':len(xs),'samples_in_episodes_with_discarded_STOP_target':sum(by_episode[e] for e in bad)}
report={'time':time.time(),'dataset':C.name,'dataset_validation_sha256':hashlib.sha256((C/'validation_report.json').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dataset_module_sha256':hashlib.sha256((S/'llava/data/dataset.py').read_bytes()).hexdigest(),
        'tokenizer_module_sha256':hashlib.sha256((S/'llava/utils/tokenizer.py').read_bytes()).hexdigest(),
        'scope':'Actual released __getitem__ text/preprocess and collator with synthetic1x1 images; no RGB validation or multimodal embedding expansion qualification.',
        'model_max_length':4096,'checked_STOP_episodes':len(records),
        'STOP_targets_fully_discarded':sum(x['all_target_tokens_discarded'] for x in records),
        'STOP_targets_partially_discarded':sum(x['target_partially_discarded'] for x in records),
        'max_input_tokens':max(x['input_tokens_before_collator'] for x in records),
        'sample_exposure':counts,'records':records,'training_data_modified':False,'model_or_GPU_allocated':False,
        'multimodal_expansion_still_requires_qualification':True}
with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
