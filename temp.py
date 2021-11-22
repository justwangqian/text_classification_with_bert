import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('data/dev.csv')
# preds = np.ones(df.shape[0])
# print(f1_score(y_true=df['labels'].values, y_pred=preds))
print(df['labels'])
print('********************************************')
print(df.labels)


log_file_name = args.log_path + args.running_mode + '.log'