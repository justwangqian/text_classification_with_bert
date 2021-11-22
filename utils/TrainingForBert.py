import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils.utils import compute_kl_loss
import pandas as pd
import numpy as np


def evaluate(model, val_dataloader, DEVICE, criterion):

    model.eval()
    val_loss = 0.0
    labels = []
    preds = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_dataloader)):
            ids = batch[0].to(DEVICE)
            att = batch[1].to(DEVICE)
            seg = batch[2].to(DEVICE)
            y = batch[3].to(DEVICE)

            logits, loss = model(ids, att, seg, y, criterion)
            val_loss += loss.item()

            y_preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            y = y.detach().cpu().numpy().tolist()

            labels.extend(y)
            preds.extend(y_preds)

        acc = accuracy_score(y_true=labels, y_pred=preds)
        F1 = f1_score(y_true=labels, y_pred=preds)

        return acc, F1, val_loss/len(val_dataloader)


def train_and_eval(model, train_dataloader, val_dataloader, DEVICE, criterion, optimizer, args, logger, scheduler=None, fold=None):

    batch_num = len(train_dataloader)
    best_acc = 0.0
    best_F1 = 0.0

    for i in range(args.EPOCHS):
        logger.info('================ Epoch {} ================='.format(i))
        # 训练一个epoch
        epoch_loss = train_one_epoch(model, train_dataloader, DEVICE, criterion, optimizer, args, scheduler)
        # 验证
        val_acc, val_F1, val_loss = evaluate(model, val_dataloader, DEVICE, criterion)
        # 更新最优指标，以及保存模型
        if val_F1 > best_F1:
            if fold is None:
                torch.save(model.state_dict(), args.model_save_path+'model.pth')
            else:
                torch.save(model.state_dict(), args.model_save_path + f'fold_{fold}_model.pth')
            logger.info(f'model saved after {i}th epoch.')

        best_acc = max(best_acc, val_acc)
        best_F1 = max(best_F1, val_F1)
        # 输出训练内容
        logger.info("val acc is {:.4f}, best acc is {:.4f}".format(val_acc, best_acc))
        logger.info("val F1 is {:.4f}, best F1 is {:.4f}".format(val_F1, best_F1))
        logger.info("train loss is {:.4f}, val loss is {:.4f}".format(epoch_loss / batch_num, val_loss))


def train_one_epoch(model, train_dataloader, DEVICE, criterion, optimizer, args, scheduler=None):

    """训练模型"""
    loss_sum = 0.0
    accumulation_step = args.accumulation
    model.train()
    for idx, batch in enumerate(tqdm(train_dataloader)):

        ids = batch[0].to(DEVICE)
        att = batch[1].to(DEVICE)
        seg = batch[2].to(DEVICE)
        y = batch[3].to(DEVICE)

        y_pred, loss = model(ids, att, seg, y, criterion)

        if args.use_R_drop:
            y_pred_2, loss_2 = model(ids, att, seg, y, criterion)
            kl_loss = compute_kl_loss(y_pred, y_pred_2)
            loss = 0.1 * kl_loss + loss_2 * 0.5 + loss * 0.5

        loss_sum += loss.item()
        loss = loss / accumulation_step
        loss.backward()

        if (idx + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

    return loss_sum


def predict(model, test_dataloader, DEVICE):

    model.eval()
    all_logits = None

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataloader)):

            ids = batch[0].to(DEVICE)
            att = batch[1].to(DEVICE)
            seg = batch[2].to(DEVICE)

            logits = model(ids, att, seg)
            # 预测概率值
            y_probs = logits.detach().cpu().numpy()
            if idx == 0:
                all_logits = y_probs
            else:
                all_logits = np.concatenate((all_logits, y_probs))

        return all_logits

