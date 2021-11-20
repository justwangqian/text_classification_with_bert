import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.Multi_sample_drop import NeuralNet as Test_Model
from dataset.DatasetForBert import CompDataset, TestDataset
import pandas as pd
from transformers import AutoTokenizer, AdamW
from utils.TrainingForBert import train_and_eval, predict
from utils.training_tricks import get_parameters
from utils.utils import compare_pinyin, clean_non_char
from utils.tools import set_args, set_logger


if __name__ == '__main__':
    args = set_args()
    logger = set_logger(args)

    logger.info("*****************  Start training   *****************")

    DEVICE = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.Model_Name, do_lower_case=True)
    # 训练集
    train_df = pd.read_csv(args.train_data_path)
    train_dataset = CompDataset(train_df, tokenizer, args.MAXLEN)
    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
    # 验证集
    val_df = pd.read_csv(args.val_data_path)
    val_dataset = CompDataset(val_df, tokenizer, args.MAXLEN)
    val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE)
    # 初始化模型
    model = Test_Model(args.Model_Name).to(DEVICE)
    # 优化器
    if args.use_multiplier:
        params = get_parameters(model=model,
                                model_init_lr=args.L_RATE,
                                multiplier=args.multiplier)
        optimizer = AdamW(params)
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.L_RATE)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练
    train_and_eval(model=model,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   DEVICE=DEVICE,
                   criterion=criterion,
                   optimizer=optimizer,
                   args=args,
                   logger=logger)
    # 加载测试数据
    test_df = pd.read_csv(args.test_data_path)
    test_dataset = TestDataset(test_df, tokenizer, args.MAXLEN)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE)
    # 加载最佳模型
    model.load_state_dict(torch.load(args.model_save_path + 'model.pth'))
    # 预测
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




