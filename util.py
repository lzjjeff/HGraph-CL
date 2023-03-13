
import argparse
from collections import OrderedDict
import torch


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mosi")
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_predict", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_type", type=str, default="bert_word")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--seeds", type=int, nargs='+', default=[123456])
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0])

    parser.add_argument("--save_path", type=str, default="./save/")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr_bert", type=float, default=5e-6)
    parser.add_argument("--lr_other", type=float, default=1e-4)
    parser.add_argument("--weight_decay_bert", type=float, default=0.001)
    parser.add_argument("--weight_decay_other", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--sup_cl_weight", type=float, default=0.1)
    parser.add_argument("--self_cl_weight", type=float, default=0.1)

    # model parameters
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--num_gnn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout_t", type=float, default=0.0)
    parser.add_argument("--dropout_v", type=float, default=0.0)
    parser.add_argument("--dropout_a", type=float, default=0.0)
    parser.add_argument("--dropout_gnn", type=float, default=0.1)
    parser.add_argument("--aug_ratio", type=float, default=0.1)

    return parser


def get_config_from_args(args):
    MODAL_SIZE = {
        "mosi": {
            'v': 20,
            'a': 5,
        },
        "mosei": {
            'v': 35,
            'a': 74
        },
    }
    bert_path_en = 'bert-base-uncased'
    bert_path_zh = 'bert-base-chinese'

    config = {}

    config["dataset"] = args.dataset
    config["batch_size"] = args.batch_size
    config["do_train"] = args.do_train
    config["do_predict"] = args.do_predict
    config["embed_type"] = args.embed_type
    config["seed"] = args.seed
    config["seeds"] = args.seeds
    config["device_ids"] = args.device_ids
    config["bert_path"] = bert_path_en

    # regression config
    config["save_path"] = args.save_path
    if not config["save_path"].endswith('/'):
        config["save_path"] += '/'

    config["max_len"] = args.max_len
    config["t_size"] = 768
    config["v_size"] = MODAL_SIZE[args.dataset]['v']
    config["a_size"] = MODAL_SIZE[args.dataset]['a']
    config["hidden_size"] = args.hidden_size
    config["num_lstm_layers"] = args.num_lstm_layers
    config["num_gnn_layers"] = args.num_gnn_layers
    config["num_gnn_heads"] = args.num_gnn_heads
    config["dropout"] = args.dropout
    config["dropout_t"] = args.dropout_t
    config["dropout_v"] = args.dropout_v
    config["dropout_a"] = args.dropout_a
    config["dropout_gnn"] = args.dropout_gnn
    config["aug_ratio"] = args.aug_ratio
    config["epoch"] = args.epoch
    config["lr_bert"] = args.lr_bert
    config["lr_other"] = args.lr_other
    config["weight_decay_bert"] = args.weight_decay_bert
    config["weight_decay_other"] = args.weight_decay_other
    config["temperature"] = args.temperature
    config["sup_cl_weight"] = args.sup_cl_weight
    config["self_cl_weight"] = args.self_cl_weight

    return config


args = get_argparse().parse_args()
config = get_config_from_args(args)

device = torch.device("cuda:%s" % config["device_ids"][0]) if torch.cuda.is_available() else "cpu"


class ResultRecoder:
    def __init__(self, name):
        self.name = name
        self.result = OrderedDict()
        self.best = {'acc7':-float('inf'), 'acc5':-float('inf'), 'non0acc2':-float('inf'),
                     'non0f1':-float('inf'), 'has0acc2':-float('inf'), 'has0f1':-float('inf'),
                     'mae':float('inf'), 'corr':-float('inf'), 'regre loss':float('inf'), 'scl loss':float('inf'),
                     'sum':-float('inf')}
        self.keys = list()

    def add(self, key, value):
        if key not in self.keys:
            self.keys.append(key)
        self.result[key] = value

    def compare_best(self, key, value):
        if key == 'mae' or key == 'regre loss' or key == 'scl loss':
            if value <= self.best[key]:
                return True
            else:
                return False
        else:
            if value >= self.best[key]:
                return True
            else:
                return False

    def update_best(self, key, value):
        self.best[key] = value

    def remove(self, key):
        self.keys.remove(key)
        self.result.pop(key)

    def output_log(self):
        log = []
        for key in self.keys:
            log.append("%s %s: %s" % (self.name, key, self.result[key]))
        return " | ".join(log)


class EpochScheduler:
    def __init__(self, patience=5, mode='min'):
        self.patience = patience
        self.wait_round = patience
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')

    def step(self, value):
        if (self.mode == 'min' and value <= self.best) or (self.mode == 'max' and value >= self.best):
            self.best = value
            self.wait_round = self.patience
            print('Current patience is: %s' % self.wait_round)
            return 1
        else:
            self.wait_round -= 1
            print('Current patience is: %s' % self.wait_round)
            if self.wait_round == 0:
                return -1    # 停止信号
            else:
                return 0   # 继续