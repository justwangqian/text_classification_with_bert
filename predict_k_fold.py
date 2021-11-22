import torch
import time
import pandas as pd
import numpy as np
from utils.TrainingForBert import predict
from model.Multi_sample_drop import NeuralNet as Test_Model
from dataset.DatasetForBert import TestDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.utils import post_process, get_time
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

    r_df = test_df.rename(columns={'text1': 'text2', 'text2': 'text1'})
    r_dataset = TestDataset(r_df, tokenizer, args.MAXLEN)
    r_dataloader = DataLoader(r_dataset, batch_size=args.BATCH_SIZE)

    model = Test_Model(model_name=args.Model_Name).to(DEVICE)

    all_logits = np.zeros((50000, 2))
    for fold in range(5):
        # 加载k折模型参数
        logger.info(f"Loading {fold} fold's model.")
        model.load_state_dict(torch.load(args.model_save_path + f'fold_{fold}_model.pth'))
        # 预测
        pred_logits = predict(model=model, test_dataloader=test_dataloader, DEVICE=DEVICE)
        all_logits = all_logits + pred_logits

        # 交换句子对顺序再次预测
        pred_logits = predict(model=model, test_dataloader=r_dataloader, DEVICE=DEVICE)
        all_logits = all_logits + pred_logits

    #计算标签
    labels = np.argmax(all_logits, axis=1)
    all_logits = pd.DataFrame(all_logits, columns=['label_0', 'label_1'])
    all_logits.to_csv(args.data_save_path + get_time() + '_all_logits.csv', index=False)
    labels = post_process(test_df, labels)
    # 保存标签文件
    pd.DataFrame(labels).to_csv(args.data_save_path + get_time() + '_labels.csv', index=False, header=None)
