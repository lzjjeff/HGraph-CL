import os
import yaml
import pickle
from tqdm import tqdm

import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup

from models.model import MultimodalGraphFusionNetwork
from util import config, device, ResultRecoder
from metrics import mosei_metrics, mosei_metrics_with_zero
from read_data import get_dataloader
from losses import SupConLoss


REGRE_MODELS = {
    "bert_word": MultimodalGraphFusionNetwork
}


def batch_fit(batch, model, loss_func, mode='train'):
    """ 模型 batch 处理,
    :param batch:
    :param model:
    :param loss_func: SupConLoss | L1Loss | [L1Loss, SupConLoss]
    :return:
    """
    batch = tuple(t.to(device) for t in batch)
    batch_size = batch[0].size(0)

    t, v, a, v_n, a_n, type, mask, adj, y, l = batch
    outputs = model(t, v, a, l, type, mask, adj)
    y_pred, reps_m, reps_t, reps_v, reps_a, reps_m_aug, reps_t_aug, reps_v_aug, reps_a_aug = outputs
    y = y.view(-1)

    if mode == 'train':
        regre_loss_func, scl_loss_func = loss_func
        regre_loss = regre_loss_func(y_pred, y)
        sup_cl_loss_m = scl_loss_func(reps_m.unsqueeze(1), y >= 0)
        sup_cl_loss_t = scl_loss_func(reps_t.unsqueeze(1), y >= 0)
        sup_cl_loss_v = scl_loss_func(reps_v.unsqueeze(1), y >= 0)
        sup_cl_loss_a = scl_loss_func(reps_a.unsqueeze(1), y >= 0)
        sup_cl_loss = sup_cl_loss_m + sup_cl_loss_t + sup_cl_loss_v + sup_cl_loss_a
        self_cl_loss_m = scl_loss_func(reps_m_aug)
        self_cl_loss_t = scl_loss_func(reps_t_aug)
        self_cl_loss_v = scl_loss_func(reps_v_aug)
        self_cl_loss_a = scl_loss_func(reps_a_aug)
        self_cl_loss = self_cl_loss_m + self_cl_loss_t + self_cl_loss_v + self_cl_loss_a
        # print(scl_loss_l.item(), scl_loss_v.item(), scl_loss_a.item())

        cl_loss = sup_cl_loss + self_cl_loss
        loss = regre_loss + sup_cl_loss * config["sup_cl_weight"] + self_cl_loss * config["self_cl_weight"]
        return loss, [regre_loss, cl_loss, sup_cl_loss, sup_cl_loss_m, sup_cl_loss_t, sup_cl_loss_v, sup_cl_loss_a,
                      self_cl_loss, self_cl_loss_m, self_cl_loss_t, self_cl_loss_v, self_cl_loss_a],\
               y_pred, y, [reps_m, reps_t, reps_v, reps_a, reps_m_aug, reps_t_aug, reps_v_aug, reps_a_aug], batch_size

    elif mode == 'test':
        regre_loss = loss_func(y_pred, y)
        return regre_loss, y_pred, y, [reps_m, reps_t, reps_v, reps_a, reps_m_aug, reps_t_aug, reps_v_aug, reps_a_aug],\
               batch_size

    else:
        raise ValueError("mode should be train or test!")


def train(train_loader, valid_loader, test_loader, model, regre_loss_func, scl_loss_func, optimizer, scheduler):
    """ 训练回归模型 """
    train_losses = []
    valid_losses = []
    last_lr = float('inf')
    train_recoder = ResultRecoder('train')
    valid_recoder = ResultRecoder('valid')
    test_recoder = ResultRecoder('test')
    optimizer_bert, optimizer_except_bert = optimizer
    scheduler_bert, scheduler_except_bert = scheduler

    if os.path.exists(os.path.join(config["save_path"], 'train_eval_result.txt')):
        os.remove(os.path.join(config["save_path"], 'train_eval_result.txt'))

    for epoch in tqdm(range(config["epoch"]), desc="回归模型训练中..."):
        # train
        lrs = []
        model.train()
        train_loss, train_regre_loss, train_cl_loss = 0.0, 0.0, 0.0
        train_sup_cl_loss, train_sup_cl_loss_m, train_sup_cl_loss_t, train_sup_cl_loss_v, train_sup_cl_loss_a = [0.0] * 5
        train_self_cl_loss, train_self_cl_loss_m, train_self_cl_loss_t, train_self_cl_loss_v, train_self_cl_loss_a = [0.0] * 5
        train_size = 0

        optimizer_except_bert.zero_grad()
        optimizer_bert.zero_grad()
        for batch in tqdm(train_loader):
            outputs = batch_fit(batch, model, [regre_loss_func, scl_loss_func], mode='train')
            loss, loss_components, y_pred, y, reps, batch_size = outputs
            regre_loss, cl_loss, sup_cl_loss, sup_cl_loss_m, sup_cl_loss_t, sup_cl_loss_v, sup_cl_loss_a,\
            self_cl_loss, self_cl_loss_m, self_cl_loss_t, self_cl_loss_v, self_cl_loss_a = loss_components
            loss.backward()
            optimizer_except_bert.step()
            optimizer_except_bert.zero_grad()
            optimizer_bert.step()
            optimizer_bert.zero_grad()
            scheduler_bert.step()

            train_size += batch_size
            train_loss += loss.item() * batch_size
            train_regre_loss += regre_loss.item() * batch_size
            train_cl_loss += cl_loss.item() * batch_size
            train_sup_cl_loss += sup_cl_loss.item() * batch_size
            train_sup_cl_loss_m += sup_cl_loss_m.item() * batch_size
            train_sup_cl_loss_t += sup_cl_loss_t.item() * batch_size
            train_sup_cl_loss_v += sup_cl_loss_v.item() * batch_size
            train_sup_cl_loss_a += sup_cl_loss_a.item() * batch_size
            train_self_cl_loss += self_cl_loss.item() * batch_size
            train_self_cl_loss_m += self_cl_loss_m.item() * batch_size
            train_self_cl_loss_t += self_cl_loss_t.item() * batch_size
            train_self_cl_loss_v += self_cl_loss_v.item() * batch_size
            train_self_cl_loss_a += self_cl_loss_a.item() * batch_size

        train_loss = train_loss / train_size
        train_regre_loss = train_regre_loss / train_size
        train_cl_loss = train_cl_loss / train_size
        train_sup_cl_loss = train_sup_cl_loss / train_size
        train_sup_cl_loss_m = train_sup_cl_loss_m / train_size
        train_sup_cl_loss_t = train_sup_cl_loss_t / train_size
        train_sup_cl_loss_v = train_sup_cl_loss_v / train_size
        train_sup_cl_loss_a = train_sup_cl_loss_a / train_size
        train_self_cl_loss = train_self_cl_loss / train_size
        train_self_cl_loss_m = train_self_cl_loss_m / train_size
        train_self_cl_loss_t = train_self_cl_loss_t / train_size
        train_self_cl_loss_v = train_self_cl_loss_v / train_size
        train_self_cl_loss_a = train_self_cl_loss_a / train_size

        train_recoder.add("loss", train_loss)
        train_recoder.add("regre loss", train_regre_loss)
        train_recoder.add("cl loss", train_cl_loss)
        train_recoder.add("sup cl loss", train_sup_cl_loss)
        train_recoder.add("sup cl loss m", train_sup_cl_loss_m)
        train_recoder.add("sup cl loss t", train_sup_cl_loss_t)
        train_recoder.add("sup cl loss v", train_sup_cl_loss_v)
        train_recoder.add("sup cl loss a", train_sup_cl_loss_a)
        train_recoder.add("self cl loss", train_self_cl_loss)
        train_recoder.add("self cl loss m", train_self_cl_loss_m)
        train_recoder.add("self cl loss t", train_self_cl_loss_t)
        train_recoder.add("self cl loss v", train_self_cl_loss_v)
        train_recoder.add("self cl loss a", train_self_cl_loss_a)

        print("EPOCH %s | %s" % (epoch, train_recoder.output_log()))

        # validation
        y_true = []
        y_pred = []
        valid_regre_loss = 0.0
        valid_size = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                outputs = batch_fit(batch, model, regre_loss_func, mode='test')
                regre_loss, _y_pred, y, reps, batch_size = outputs
                valid_regre_loss += regre_loss.item() * batch_size
                valid_size += batch_size
                y_true.append(y.cpu())
                y_pred.append(_y_pred.cpu())

        valid_regre_loss = valid_regre_loss / valid_size

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        acc7, acc5, acc2, f1, mae, corr = mosei_metrics(y_true, y_pred)
        _acc7, _acc5, _acc2, _f1, _mae, _corr = mosei_metrics_with_zero(y_true, y_pred, config["dataset"])

        valid_recoder.add("regre loss", valid_regre_loss)

        valid_recoder.add("acc7", acc7)
        valid_recoder.add("acc5", acc5)
        valid_recoder.add("non0acc2", acc2)
        valid_recoder.add("non0f1", f1)
        valid_recoder.add("has0acc2", _acc2)
        valid_recoder.add("has0f1", _f1)
        valid_recoder.add("mae", mae)
        valid_recoder.add("corr", corr)

        print("EPOCH %s | %s" % (epoch, valid_recoder.output_log()))

        # 保存结果
        with open(os.path.join(config["save_path"], 'train_eval_result.txt'),
                  'a', encoding='utf-8') as fo:
            fo.write("\nEPOCH {0} | {1}"
                     "\nEPOCH {0} | {2}"
                     "\nEPOCH {0} | Current bert learning rate: {3}"
                     "\nEPOCH {0} | Current other learning rate: {4}".format(
                epoch, train_recoder.output_log(), valid_recoder.output_log(),
                optimizer_bert.state_dict()['param_groups'][0]['lr'],
                optimizer_except_bert.state_dict()['param_groups'][0]['lr']))

        # 保存模型
        if valid_recoder.compare_best('sum', acc7+acc5+f1+_f1-mae+corr):
            valid_recoder.update_best('sum', acc7+acc5+f1+_f1-mae+corr)
            print("A new best model on valid set")
            if device == "cpu":
                torch.save(model.state_dict(), os.path.join(config["save_path"], 'model.std'))
            else:
                torch.save(model.module.state_dict(), os.path.join(config["save_path"], 'model.std'))
            print("Current bert learning rate: %s" % optimizer_bert.state_dict()['param_groups'][0]['lr'])
            print("Current other learning rate: %s" % optimizer_except_bert.state_dict()['param_groups'][0]['lr'])

        # 更新学习率并读取最佳模型
        scheduler_except_bert.step(valid_regre_loss)
        if optimizer_except_bert.state_dict()['param_groups'][0]['lr'] < last_lr:
            if device == "cpu":
                model.load_state_dict(torch.load(os.path.join(config["save_path"], 'model.std')))
            else:
                model.module.load_state_dict(torch.load(os.path.join(config["save_path"], 'model.std')))
            last_lr = optimizer_except_bert.state_dict()['param_groups'][0]['lr']


def predict(test_loader, model, regre_loss_func):
    """ 在测试集上进行预测并输出结果 """
    y_true = []
    y_pred = []
    test_recoder = ResultRecoder('test')

    model.eval()
    with torch.no_grad():
        test_regre_loss = 0.0
        test_size = 0
        for batch in tqdm(test_loader):
            outputs = batch_fit(batch, model, regre_loss_func, mode='test')
            regre_loss, _y_pred, y, reps, batch_size = outputs
            y_true.append(y.cpu())
            y_pred.append(_y_pred.cpu())
            test_regre_loss += regre_loss.item() * batch_size
            test_size += batch_size

            reps_m, reps_t, reps_v, reps_a, reps_m_aug, reps_t_aug, reps_v_aug, reps_a_aug = reps

    print(test_regre_loss, test_size)
    test_regre_loss = test_regre_loss / test_size

    test_recoder.add("regre loss", test_regre_loss)

    print(test_recoder.output_log())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    acc7, acc5, acc2, f1, mae, corr = mosei_metrics(y_true, y_pred)
    print("\nTest set acc7 is %s"
          "\nTest set acc5 is %s"
          "\nTest set acc2 is %s"
          "\nTest set f1 score is %s"
          "\nTest set MAE is %s"
          "\nTest set Corr is %s"
          % (acc7, acc5, acc2, f1, mae, corr))

    # 保存结果
    with open(os.path.join(config["save_path"], 'predict_result.txt'),
              'a', encoding='utf-8') as fo:
        fo.write("\nTest loss: %s\nTest set acc7 is %s\nTest set acc5 is %s\nTest set acc2 is %s\n"
                 "Test set f1 score is %s\nTest set MAE is %s\nTest set Corr is %s\n"
                 % (round(test_regre_loss, 4), acc7, acc5, acc2, f1, mae, corr))

    _acc7, _acc5, _acc2, _f1, _mae, _corr = mosei_metrics_with_zero(y_true, y_pred, config["dataset"])
    print("\nTest set acc7 with zero is %s"
          "\nTest set acc5 with zero is %s"
          "\nTest set acc2 with zero is %s"
          "\nTest set f1 score with zero is %s"
          "\nTest set MAE is %s"
          "\nTest set Corr is %s"
          % (_acc7, _acc5, _acc2, _f1, _mae, _corr))

    # 保存结果
    with open(os.path.join(config["save_path"], 'predict_result_with_zero.txt'),
              'a', encoding='utf-8') as fo:
        fo.write("\nTest loss: %s\nTest set acc_7 is %s\nTest set acc_5 is %s\nTest set acc2 is %s\n"
                 "Test set f1 score is %s\nTest set MAE is %s\nTest set Corr is %s\n"
                 % (round(test_regre_loss, 4), _acc7, _acc5, _acc2, _f1, _mae, _corr))

    # 保存预测结果
    pd.DataFrame({"pred": y_pred, "true": y_true}).to_csv(os.path.join(config["save_path"], 'predict_score.csv'))


def run_msa(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 载入数据
    print("载入数据...")
    train_loader, valid_loader, test_loader = get_dataloader(config)

    # 检查输出目录
    if not os.path.exists(config["save_path"]):
        os.mkdir(config["save_path"])
    # 保存参数文件
    with open('{}/config.yaml'.format(config["save_path"]), 'w') as f:
        yaml.dump(config, f)

    scl_loss_func = SupConLoss(temperature=config["temperature"], base_temperature=config["temperature"]) 
    regresson_loss_func = nn.L1Loss()

    if config["do_train"]:
        # 创建回归模型
        print("创建回归模型...")
        regression_model = REGRE_MODELS[config["embed_type"]](config)
        if not device == "cpu":
            regression_model = nn.DataParallel(regression_model, device_ids=config["device_ids"])
        regression_model = regression_model.to(device)

        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(regression_model.module.encoder_t.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        other_params = [p for n, p in list(regression_model.module.named_parameters()) if 'encoder_t' not in n]

        optimizer_bert_parameters = [
            {'params': bert_params_decay, 'weight_decay': config["weight_decay_bert"], 'lr': config["lr_bert"]},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': config["lr_bert"]}
        ]
        optimizer_parameters_except_bert = [
            {'params': other_params, 'weight_decay': config["weight_decay_other"], 'lr': config["lr_other"]}
        ]
        optimizer_bert = optim.Adam(optimizer_bert_parameters)
        optimizer_except_bert = optim.Adam(optimizer_parameters_except_bert)

        total_steps = len(train_loader) * config["epoch"]
        scheduler_bert = get_cosine_schedule_with_warmup(optimizer_bert, num_warmup_steps=0.1 * total_steps,
                                                         num_training_steps=total_steps)
        scheduler_except_bert = ReduceLROnPlateau(optimizer_except_bert, mode='min', patience=5, factor=0.1, verbose=True)

        print("训练编码器...")
        train(train_loader, valid_loader, test_loader, regression_model, regresson_loss_func, scl_loss_func,
                  [optimizer_bert, optimizer_except_bert],
                  [scheduler_bert, scheduler_except_bert])

    if config["do_predict"]:
        regression_model = REGRE_MODELS[config["embed_type"]](config)
        if not device == "cpu":
            regression_model = nn.DataParallel(regression_model, device_ids=config["device_ids"])
        regression_model = regression_model.to(device)

        if device == "cpu":
            regression_model.load_state_dict(torch.load(os.path.join(config["save_path"], 'model.std')))
        else:
            regression_model.module.load_state_dict(torch.load(os.path.join(config["save_path"], 'model.std')))

        predict(test_loader, regression_model, regresson_loss_func)


if __name__ == '__main__':
    root_save_path = config["save_path"]
    if not os.path.exists(root_save_path):
        os.mkdir(root_save_path)
    
    for seed in config["seeds"]:
        config["save_path"] = os.path.join(root_save_path, '%s' % seed)
        print("Current seed: %s" % seed)
        print("Configs: ", config)
        run_msa(seed)
