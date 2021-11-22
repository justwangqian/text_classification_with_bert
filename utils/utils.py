import time

from pypinyin import lazy_pinyin
import re
import torch.nn.functional as F


# 计算两个离散概率分布之间的KL散度
def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


# 比较两个中文文本的拼音是否相同
def compare_pinyin(s1, s2):
    s1_pinyin = ''.join(lazy_pinyin(s1))
    s2_pinyin = ''.join(lazy_pinyin(s2))
    return s1_pinyin == s2_pinyin


# 清除句子中的标点符号
def clean_non_char(text):
    return re.sub('[^0-9a-zA-Z\u4e00-\u9fa5]+', '', text)


# 检查拼音是否相同
def post_process(test_df, labels):
    for i, row in test_df.iterrows():
        # 清除标点后句子拼音相同则设置标签为1
        if compare_pinyin(clean_non_char(row['text1']), clean_non_char(row['text2'])):
            labels[i] = 1
    return labels


# 获取当前时间
def get_time():
    return time.strftime("%Y-%m-%d-%X", time.localtime())
