import torch
import time
import pandas as pd
from utils.TrainingForBert import predict
from model.text_cnn import Text_CNN_Bert as Test_Model
from dataset.DatasetForBert import TestDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.utils import compare_pinyin, clean_non_char
from utils.tools import set_args, set_logger

if __name__ == '__main__':
    args = set_args()
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

    all_logits = [[0, 0]] * 50000
    for i in range(4):
        _, pred_logits = predict(model=model,
                                 test_dataloader=test_dataloader,
                                 DEVICE=DEVICE)
        # 保存logits
        logits_df = pd.DataFrame(pred_logits, columns=['label_0', 'label_1'])
        local_time = time.strftime("%Y-%m-%d-%X", time.localtime())
        logits_df.to_csv(args.data_save_path + local_time + f'_{i}_logits.csv', index=False)

        for j in range(50000):
            all_logits[i][0] += pred_logits[i][0]
            all_logits[i][1] += pred_logits[i][1]

        _, pred_logits = predict(model=model,
                                 test_dataloader=r_dataloader,
                                 DEVICE=DEVICE)
        # 保存logits
        logits_df = pd.DataFrame(pred_logits, columns=['label_0', 'label_1'])
        local_time = time.strftime("%Y-%m-%d-%X", time.localtime())
        logits_df.to_csv(args.data_save_path + local_time + f'_r_{i}_logits.csv', index=False)

        for j in range(50000):
            all_logits[i][0] += pred_logits[i][0]
            all_logits[i][1] += pred_logits[i][1]

    labels = []
    for i in range(50000):
        if all_logits[i][0] > all_logits[i][1]:
            labels.append(0)
        else:
            labels.append(1)

    all_logits = pd.DataFrame(all_logits, columns=['label_0', 'label_1'])
    local_time = time.strftime("%Y-%m-%d-%X", time.localtime())
    all_logits.to_csv(args.data_save_path + local_time + '_all_logits.csv', index=False)

    # 检查拼音是否相同
    for i, row in test_df.iterrows():
        # 清除标点后句子拼音相同则设置标签为1
        if compare_pinyin(clean_non_char(row['text1']), clean_non_char(row['text2'])):
            labels[i] = 1
    # 保存标签文件
    pd.DataFrame(labels).to_csv(args.data_save_path + local_time + '_labels.csv',
                                index=False,
                                header=None)
