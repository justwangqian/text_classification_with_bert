import torch
import warnings
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.multi_sample_dropout import NeuralNet as Test_Model
from dataset.dataset import CompDataset, TestDataset
import pandas as pd
from transformers import AutoTokenizer, AdamW
from utils.train_eval_infer import train_and_eval, evaluate
from utils.training_tricks import get_parameters
from utils.utils import compare_pinyin
from utils.train_eval_infer import predict

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/train.csv', type=str, required=False, help='训练数据路径')
    parser.add_argument('--val_data_path', default='data/all_dev.csv', type=str, required=False, help='验证数据路径')
    parser.add_argument('--test_data_path', default='data/test.csv', type=str, required=False, help='测试数据路径')
    parser.add_argument('--MAXLEN', default=64, type=int, required=False, help='输入文本的最大保留长度')
    parser.add_argument('--log_path', default='log/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--EPOCHS', default=10, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--BATCH_SIZE', default=256, type=int, required=False, help='训练的batch size')
    parser.add_argument('--L_RATE', default=2e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--accumulation', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪后的最大梯度')
    parser.add_argument('--model_save_path', default='model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--Model_Name', default="peterchou/ernie-gram", type=str, required=False, help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.")
    parser.add_argument('--warmup_radio', type=float, default=0.1, help='warm_up占总更新次数的比例')
    parser.add_argument('--num_class', type=int, default=2, help='标签的分类数')
    parser.add_argument('--multiplier', type=float, default=0.95, help='模型内部学习率衰减速度')
    parser.add_argument('--use_R_drop', type=bool, default=False, help='是否使用R-drop')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.Model_Name, do_lower_case=True)

    train_df = pd.read_csv(args.train_data_path)
    train_dataset = CompDataset(train_df, tokenizer, args.MAXLEN)
    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)

    val_df = pd.read_csv(args.val_data_path)
    val_dataset = CompDataset(val_df, tokenizer, args.MAXLEN)
    val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE)

    model = Test_Model(args.Model_Name).to(DEVICE)

    params = get_parameters(model=model,
                            model_init_lr=args.L_RATE,
                            multiplier=args.multiplier)
    optimizer = AdamW(params)

    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.L_RATE)
    criterion = nn.CrossEntropyLoss()

    train_and_eval(model=model,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   DEVICE=DEVICE,
                   criterion=criterion,
                   optimizer=optimizer,
                   args=args)

    # acc, F1, loss = evaluate(model, val_dataloader, DEVICE, criterion)
    # print('acc: ', acc)
    # print('F1: ', F1)
    # print('loss； ', loss)

    test_df = pd.read_csv(args.test_data_path)
    test_dataset = TestDataset(test_df, tokenizer, args.MAXLEN)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)

    model = Test_Model(model_name=args.Model_Name).to(DEVICE)
    model.load_state_dict(torch.load(args.model_save_path + 'model.pth'))

    preds = predict(model=model,
                    test_dataloader=test_dataloader,
                    DEVICE=DEVICE)

    for i, row in test_df.iterrows():
        if compare_pinyin(row['text1'], row['text2']):
            preds[i] = 1

    pd.DataFrame(preds).to_csv('data/ccf_qianyan_qm_result_A.csv', index=False, header=None)




