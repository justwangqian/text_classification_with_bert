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

    model = Test_Model(model_name=args.Model_Name).to(DEVICE)
    model.load_state_dict(torch.load(args.model_save_path+'fold_1_model.pth'))

    pred_labels, pred_logits = predict(model=model,
                                       test_dataloader=test_dataloader,
                                       DEVICE=DEVICE)
    # 保存logits
    logits_df = pd.DataFrame(pred_logits, columns=['label_0', 'label_1'])
    local_time = time.strftime("%Y-%m-%d-%X", time.localtime())
    logits_df.to_csv(args.data_save_path + local_time + '_logits.csv', index=False)
    # 检查拼音是否相同
    for i, row in test_df.iterrows():
        # 清除标点后句子拼音相同则设置标签为1
        if compare_pinyin(clean_non_char(row['text1']), clean_non_char(row['text2'])):
            pred_labels[i] = 1
    # 保存标签文件
    pd.DataFrame(pred_labels).to_csv(args.data_save_path + local_time + '_labels.csv',
                                     index=False,
                                     header=None)
