import pandas as pd
from utils.utils import compare_pinyin
from tqdm import tqdm
import torch


data = []

with open('data/dev', 'r') as f:
    lines = f.readlines()
    for line in lines:
        sep = line.strip().split('\t')
        data.append(sep)

df = pd.DataFrame(data, columns=['text1', 'text2', 'labels'])
df.to_csv('dev.csv', index=False)



