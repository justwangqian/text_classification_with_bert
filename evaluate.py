import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer
from model.Multi_sample_drop import NeuralNet as Test_Model
from dataset.DatasetForBert import CompDataset
from utils.TrainingForBert import evaluate
from utils.tools import set_args, set_logger


if __name__ == '__main__':

    args = set_args()
    args.running_mode = 'val'
    logger = set_logger(args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.Model_Name, do_lower_case=True)

    # 验证集
    val_df = pd.read_csv(args.val_data_path)
    val_dataset = CompDataset(val_df, tokenizer, args.MAXLEN)
    val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE)
    # 初始化模型
    model = Test_Model(args.Model_Name).to(DEVICE)
    model.load_state_dict(torch.load(args.model_save_path + 'model.pth'))
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    acc, F1, loss = evaluate(model, val_dataloader, DEVICE, criterion)
    logger.info(f'acc: {acc}')
    logger.info(f'F1: {F1}')
    logger.info(f'loss: {loss}')

