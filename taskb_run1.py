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


Repo.clone_from("https://huggingface.co/dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb","main") 
os.rename("main","mdl")

tokenizer = AutoTokenizer.from_pretrained('mdl')
model = AutoModelForSeq2SeqLM.from_pretrained('mdl',
                                              trust_remote_code=True).to(device='cuda:0')

preds = []
for batch in tqdm(dataloader):
    inp = tokenizer(batch,max_length=3652,padding="max_length",
                    truncation=True,return_tensors='pt').to(device='cuda:0')
    res = model.generate(**inp,max_length=1291)
    res = res.to(device='cpu')
    outputs = tokenizer.batch_decode(res,skip_special_tokens=True)
    preds.append(outputs)

resolve_list = lambda x:[j for i in x for j in i]

df = pd.read_csv(sys.argv[1])
df['SystemOutput'] = resolve_list(preds)
df['SystemOutput'] = df['SystemOutput'].apply(lambda x:re.sub("[^\w\d\s\/:\[\]\.\-\,]","",x)) 

tokenizer = None
model = None
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

shutil.rmtree('mdl')

if not os.path.exists('outputs'):
   os.mkdir('outputs')

df = df[['encounter_id','SystemOutput']]
df.columns = ['TestID','SystemOutput']
df.to_csv('outputs/taskB_iuteam1_run1.csv',index=False)