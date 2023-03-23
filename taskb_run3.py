import os
import re
import gc
import sys
import nltk
import torch
import shutil
import pandas as pd

from git import Repo
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')

dataset = load_dataset("csv", data_files={"test": sys.argv[1]})
dataloader = torch.utils.data.DataLoader(dataset['test']['dialogue'], 
                                         batch_size = 4)

headers = ['CHIEF COMPLAINT', 'HISTORY OF PRESENT ILLNESS', 'PAST MEDICAL FAMILY AND SOCIAL HISTORY', 'PHYSICAL EXAM', 'RESULTS', 'ASSESSMENT AND PLAN']

ml_dict =  {'CHIEF COMPLAINT':128,
            'HISTORY OF PRESENT ILLNESS':535,
            'PAST MEDICAL FAMILY AND SOCIAL HISTORY':206,
            'PHYSICAL EXAM':286,
            'RESULTS':217,
            'ASSESSMENT AND PLAN':548}

def dedup_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

for i in headers:
  j = i.replace(" ","_").lower()
  Repo.clone_from("https://huggingface.co/dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb-sectionwise-"+j,"main") 
  os.rename("main",j)

sec_preds = {}
for i in headers:
  model_id=i.replace(" ","_").lower()

  tokenizer = AutoTokenizer.from_pretrained(model_id,revision="v1.4")
  model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                              revision="v1.4",
                                              trust_remote_code=True).to(device='cuda:0')
  
  preds = []
  for batch in tqdm(dataloader):
      inp = tokenizer(batch,max_length=3652,padding="max_length",
                      truncation=True,return_tensors='pt').to(device='cuda:0')
      res = model.generate(**inp,max_length=ml_dict[model_id.replace("_"," ").upper()])
      res = res.to(device='cpu')
      outputs = tokenizer.batch_decode(res,skip_special_tokens=True)
      preds.append(outputs)
  sec_preds[i] = preds
  
  shutil.rmtree(model_id)

  tokenizer = None
  model = None
  gc.collect()
  with torch.no_grad():
      torch.cuda.empty_cache()

resolve_list = lambda x:[j for i in x for j in i]
sec_preds_proc = {k:resolve_list(v) for k,v in sec_preds.items()}

df1 = pd.DataFrame.from_dict(sec_preds_proc)
df2 = pd.read_csv(sys.argv[1])

post_proc = lambda x: "\n\n".join(dedup_list(sent_tokenize(x)))
for i in headers:
  df1[i] = df1[i].apply(post_proc)

df1['Output'] = df1.apply(lambda x:"\n\n".join([i+"\n\n"+x[i] for i in headers]),axis=1)
df1['Output'] = df1['Output'].apply(lambda x:re.sub("[^\w\d\s\/:\[\]\.\-\,]","",x)) 

Repo.clone_from("https://huggingface.co/dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb-sectionwise-combiner-base","main") 
os.rename("main","combiner")

df1.to_csv('temp.csv',index=False)

combiner_dataset = load_dataset('csv',data_files={'test':'temp.csv'})
combiner_dataloader = torch.utils.data.DataLoader(combiner_dataset['test']['Output'], 
                                         batch_size = 4)

os.remove('temp.csv')

tokenizer = AutoTokenizer.from_pretrained('combiner',revision="v1.1")
model = AutoModelForSeq2SeqLM.from_pretrained('combiner',
                                              revision="v1.1",
                                              trust_remote_code=True).to(device='cuda:0')

preds = []
for batch in tqdm(combiner_dataloader):
    inp = tokenizer(batch,max_length=1405,padding="max_length",
                    truncation=True,return_tensors='pt').to(device='cuda:0')
    res = model.generate(**inp,max_length=1291)
    res = res.to(device='cpu')
    outputs = tokenizer.batch_decode(res,skip_special_tokens=True)
    preds.append(outputs)

df2['SystemOutput'] = resolve_list(preds)
df2['SystemOutput'] = df2['SystemOutput'].apply(lambda x:re.sub("[^\w\d\s\/:\[\]\.\-\,]","",x)) 

tokenizer = None
model = None
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

shutil.rmtree('combiner')

if not os.path.exists('outputs'):
   os.mkdir('outputs')

df2 = df2[['encounter_id','SystemOutput']]
df2.columns = ['TestID','SystemOutput']
df2.to_csv('outputs/taskB_iuteam1_run3.csv',index=False)