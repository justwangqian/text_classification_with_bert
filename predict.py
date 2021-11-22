import numpy as np
import torch
import time
import pandas as pd
from utils.TrainingForBert import predict
from model.Multi_sample_drop import NeuralNet as Test_Model
from dataset.DatasetForBert import TestDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.utils import get_time, post_process
from utils.tools import set_args, set_logger


if __name__ == '__main__':
    args = set_args()
    args.running_mode = 'predict'
    logger = set_logger(args)

    logger.info("*****************  Start predicting   *****************")

    DEVICE = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.Model_Name, do_lower_case=True)

    test_df = pd.read_csv(args.test_data_path)
    test_dataset = TestDataset(test_df, tokenizer, args.MAXLEN)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)

    model = Test_Model(model_name=args.Model_Name).to(DEVICE)
    model.load_state_dict(torch.load(args.model_save_path+'fold_1_model.pth'))

    pred_logits = predict(model=model, test_dataloader=test_dataloader, DEVICE=DEVICE)
    # 保存logits
    logits_df = pd.DataFrame(pred_logits, columns=['label_0', 'label_1'])
    logits_df.to_csv(args.data_save_path + get_time() + '_logits.csv', index=False)
    labels = post_process(test_df, np.argmax(pred_logits, axis=1))
    # 保存标签文件
    pd.DataFrame(labels).to_csv(args.data_save_path + get_time() + '_labels.csv', index=False, header=None)
