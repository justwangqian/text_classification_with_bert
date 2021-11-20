import argparse
import logging


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='data/all.csv', type=str, required=False, help='训练数据路径')
    parser.add_argument('--test_data_path', default='data/test.csv', type=str, required=False, help='测试数据路径')
    parser.add_argument('--gpu_index', default=0, type=int, required=False, help='测试数据路径')
    parser.add_argument('--MAXLEN', default=64, type=int, required=False, help='输入文本的最大保留长度')
    parser.add_argument('--log_path', default='log/train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='对于ignore_index的label token不计算梯度')
    parser.add_argument('--EPOCHS', default=7, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--BATCH_SIZE', default=256, type=int, required=False, help='训练的batch size')
    parser.add_argument('--L_RATE', default=2e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='AdamW优化器的衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--accumulation', default=1, type=int, required=False, help='梯度积累的步数')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪后的最大梯度')
    parser.add_argument('--model_save_path', default='saved_model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--data_save_path', default='predicted_data/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--Model_Name', default="peterchou/ernie-gram", type=str, required=False, help='预训练的模型的路径')
    parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.")
    parser.add_argument('--warmup_radio', type=float, default=0.1, help='warm_up占总更新次数的比例')
    parser.add_argument('--num_class', type=int, default=2, help='标签的分类数')
    parser.add_argument('--use_multiplier', type=bool, default=True, help='模型内部学习率衰减速度')
    parser.add_argument('--multiplier', type=float, default=0.95, help='模型内部学习率衰减速度')
    parser.add_argument('--use_R_drop', type=bool, default=False, help='是否使用R-drop')
    parser.add_argument('--K_fold', type=int, default=5, help='将数据分成K折')
    args = parser.parse_args()
    return args


def set_stream_handler():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    return handler


def set_file_handler(file_path, mode='a'):
    if mode != 'a' or mode != 'w':
        mode = 'a'
    handler = logging.FileHandler(file_path, mode=mode)
    handler.setLevel(logging.DEBUG)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_fmt = "%m/%d/%Y %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    handler.setFormatter(formatter)
    return handler


def set_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(set_file_handler(args.log_path))
    logger.addHandler(set_stream_handler())
    logger.debug('------------Start running------------')
    logger.debug('Model_Name : {}'.format(args.Model_Name))
    logger.debug('Batch_size : {}'.format(args.BATCH_SIZE))
    logger.debug('L_RATE : {}'.format(args.L_RATE))
    logger.debug('MAXLEN : {}'.format(args.MAXLEN))
    logger.debug('accumulation : {}'.format(args.accumulation))
    logger.debug('use_multiplier : {}'.format(args.use_multiplier))
    logger.debug('multiplier : {}'.format(args.multiplier))
    logger.debug('use_R_drop : {}'.format(args.use_R_drop))
    return logger
