# from torch.utils.data import DataLoader, Dataset
import torch.utils.data
from transformers import T5Tokenizer, T5ForConditionalGeneration
#import sentencepiece as spm
import requests
import pandas as pd

model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
token = T5Tokenizer.from_pretrained(model_name)

url = "http://dibresources.jcbose.ac.in/ssaha4/pulmopred/public/training.csv.txt"
response = requests.get(url)
#data = response.text
ds = pd.read_csv(url)
ds = ds[~ds['Diagnosis'].isin(['DPLD', 'OSA', 'Sarcodiosis','Chest Pain'])]
ds = ds.drop(columns = ['Label'])
ds.dropna(inplace=True)

#ds = pd.read_csv(pd.compat.StringIO(data))

#diagnoses_to_exclude = ['DPLD', 'OSA', 'Sarcodiosis', 'Chest Pain']
#ds_filtered = ds[~ds['Diagnosis'].isin(diagnoses_to_exclude)]

#ds_filtered = ds_filtered.drop(columns=['Label'])

#ds_filtered.dropna(inplace=True)