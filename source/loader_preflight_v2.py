"""Exercise real AeroAct dataset/collator; verify identical paired model inputs."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, tempfile, time, traceback
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--source',required=True);ap.add_argument('--dataset',default='complete-proxy-pass-metric-v2');a=ap.parse_args()
R=Path(a.root).resolve();S=Path(a.source).resolve();C=R/'dataset'/a.dataset
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(S))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
try:
 import torch
 from transformers import AutoTokenizer,SiglipImageProcessor,set_seed
 from llava.train.args import DataArguments,TrainingArguments
 from llava.data.dataset import LazyUpdateTripleMergeAirVLNDataset,DataCollatorForSupervisedDataset
 from llava import conversation as conv
 torch.set_num_threads(4)
 V=json.loads((C/'validation_report.json').read_text())
 M=Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official/runtime/models/NVILA-Lite-8B-ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d')
 tok=AutoTokenizer.from_pretrained(str(M/'llm'),model_max_length=4096,padding_side='right',use_fast=False,legacy=False);conv.default_conversation=conv.conv_templates['auto']
 processor=SiglipImageProcessor.from_pretrained(str(M/'vision_tower'))
 original=LazyUpdateTripleMergeAirVLNDataset._load_video
 def checked(*args,**kwargs):
  images,valid=original(*args,**kwargs);assert valid,'No empty-image fallback permitted';return images,valid
 LazyUpdateTripleMergeAirVLNDataset._load_video=staticmethod(checked)
 datasets={};reports={}
 for name in V['sample_counts']:
  D=C/name;P=Path(tempfile.mkdtemp(prefix='aerialvln_control_loader_'));dd=P/'Dataset/AerialVLN-Dataset/data';dd.mkdir(parents=True);(dd/'aerialvln-s').symlink_to(D/'data/aerialvln-s',target_is_directory=True);os.chdir(P)
  da=DataArguments(image_aspect_ratio='dynamic',num_video_frames=8,idx_type='merge',hisotry_queue_len=0,temporal_aggregation_frames=8,label_balance=True,label_balance_type='v1',is_multimodal=True);da.image_processor=processor;da.prob_stop=0.0
  ta=TrainingArguments(output_dir=str(P/'output'),bf16=True,per_device_train_batch_size=4,gradient_accumulation_steps=2)
  set_seed(10);ds=LazyUpdateTripleMergeAirVLNDataset(str(D/'data/aerialvln-s/train_merged_triple.json'),str(D/'Raw_data/aerialvln-s'),tok,da,ta);assert len(ds)==V['sample_counts'][name] and len(ds.ids_merge)==V['usable_instructions']
  reps={}
  for i,row in enumerate(ds.list_data_dict):reps.setdefault(row['action'],i)
  assert set(reps)==set(V['action_text_counts'][name]), 'Exercise every action class present in this frozen group'
  collator=DataCollatorForSupervisedDataset(tokenizer=tok,data_args=da)
  examples=[]
  for act,i in reps.items():
   x=ds[i];assert tuple(x['image'].shape)==(8,3,448,448) and torch.isfinite(x['image']).all();assert (x['labels']!=-100).any()
   batch=collator([x]);assert 'sample_weight' not in batch,'Strict release-collator behavior must hold.'
   examples.append({'action':act,'supervised_tokens':int((x['labels']!=-100).sum())})
  reports[name]={'loaded_samples':len(ds),'episodes':len(ds.ids_merge),'tested_classes':len(reps),'examples':examples,'sample_weight_in_collated_batch':False,'RGB_decode_fallback_allowed':False}
  datasets[name]=ds
  print('LOADER_GROUP_PASSED',name,len(ds),flush=True)
 left=datasets['gt-joint-boundary'];right=datasets['rec-joint-boundary']
 assert left.ids_merge==right.ids_merge and left.episode2instruction==right.episode2instruction
 assert len(left)==len(right)
 changes=[];selected=[];seen=set()
 for i,(x,y) in enumerate(zip(left.list_data_dict,right.list_data_dict)):
  assert {k:v for k,v in x.items() if k!='action'}=={k:v for k,v in y.items() if k!='action'}
  signature=(x['action'],y['action'])
  if signature not in seen and (x['action']!=y['action'] or 'stop.' in x['action']):selected.append(i);seen.add(signature)
  if x['action']!=y['action']:changes.append(i)
  if len(selected)>=24:break
 checks=[]
 for i in selected:
  x,y=left[i],right[i]
  assert torch.equal(x['image'],y['image'])
  nx=int(torch.nonzero(x['labels']!=-100)[0]);ny=int(torch.nonzero(y['labels']!=-100)[0])
  assert nx==ny and torch.equal(x['input_ids'][:nx],y['input_ids'][:ny])
  row=left.list_data_dict[i];checks.append({'episode':row['episode'],'timestep':row['timestep'],'GT':row['action'],'REC':right.list_data_dict[i]['action'],'exact_image_tensor_equal':True,'exact_prompt_prefix_equal':True})
 report={'passed':True,'time':time.time(),'source':str(S),'source_dataset_py_sha256':hashlib.sha256((S/'llava/data/dataset.py').read_bytes()).hexdigest(),'groups':reports,'joint_boundary_real_tensor_prompt_checks':checks,'joint_boundary_pair_same_inputs_confirmed':True,'physical_legality_certified':False,'navigation_evaluation_performed':False,'model_training_performed':False}
 proof=R/'report'/('official_loader_preflight_'+a.dataset+'.json');write(proof,report)
 write(C/'READY.json',{'status':'ready','scope':'Actual loader structural checks only; optical-screened research subset, physical legality unproven; formal training review separate.','validation_report_sha256':hashlib.sha256((C/'validation_report.json').read_bytes()).hexdigest(),'official_loader_preflight_sha256':hashlib.sha256(proof.read_bytes()).hexdigest()})
 print('CONTROLLED_DATASET_READY',flush=True)
except BaseException:
 write(R/'report'/('official_loader_preflight_failure_'+a.dataset+'.json'),{'time':time.time(),'traceback':traceback.format_exc()});raise
