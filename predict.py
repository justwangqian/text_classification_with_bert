import torch
import pandas as pd
from utils.train_eval_infer import predict
from model.text_cnn import Text_CNN_Bert as Test_Model
from run import set_args
from dataset.dataset import TestDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from utils.utils import compare_pinyin


if __name__ == '__main__':
    args = set_args()
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.Model_Name)

    test_df = pd.read_csv(args.test_data_path)
    test_dataset = TestDataset(test_df, tokenizer, args.MAXLEN)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)

    model = Test_Model(model_name=args.Model_Name).to(DEVICE)
    model.load_state_dict(torch.load(args.model_save_path+'model.pth'))

    preds = predict(model=model,
                    test_dataloader=test_dataloader,
                    DEVICE=DEVICE)

    for i, row in test_df.iterrows():
        if compare_pinyin(row['text1'], row['text2']):
            preds[i] = 1

    pd.DataFrame(preds).to_csv('data/ccf_qianyan_qm_result_A.csv', index=False, header=None)
