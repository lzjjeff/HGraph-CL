import pickle
import copy
import numpy as np
import torch.utils.data as Data
from dataset import MoseiDataset, collate_fn


def get_dataloader(config):
    if config["dataset"] == "mosi":
        if config["embed_type"] == 'bert_word':
            with open('./data/mosi_word_aligned_50.dataset', 'rb') as f:
                data_dic = pickle.load(f)
        else:
            raise ValueError("Invalid embed_type !")
    elif config["dataset"] == "mosei":
        if config["embed_type"] == 'bert_word':
            with open('./data/mosei_word.dataset', 'rb') as f:
                data_dic = pickle.load(f)
        else:
            raise ValueError("Invalid embed_type !")
    else:
        raise ValueError("Invalid dataset !")

    # replace inf and nan by 0
    for dname in data_dic.keys():
        for key in data_dic[dname].keys():
            v = copy.deepcopy(data_dic[dname][key]["v"])
            a = copy.deepcopy(data_dic[dname][key]["a"])
            v[~np.isfinite(v)] = 0
            a[~np.isfinite(a)] = 0
            data_dic[dname][key]["v"] = v
            data_dic[dname][key]["a"] = a

    # normalize
    v_max = np.max(
        np.array([np.max(np.abs(data_dic['train'][key]['v']), axis=0) for key in data_dic['train'].keys()]), axis=0)
    a_max = np.max(
        np.array([np.max(np.abs(data_dic['train'][key]['a']), axis=0) for key in data_dic['train'].keys()]), axis=0)
    v_max[v_max == 0] = 1
    a_max[a_max == 0] = 1

    train_dataset = MoseiDataset(data_dic, 'train', embed_type=config["embed_type"], v_max=v_max, a_max=a_max)
    valid_dataset = MoseiDataset(data_dic, 'valid', embed_type=config["embed_type"], v_max=v_max, a_max=a_max)
    test_dataset = MoseiDataset(data_dic, 'test', embed_type=config["embed_type"], v_max=v_max, a_max=a_max)

    train_loader = Data.DataLoader(train_dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    print("\tNumber of train samples: %d\n\tNumber of valid samples: %d\n\tNumber of test samples: %d"
          % (len(train_dataset), len(valid_dataset), len(test_dataset)))

    return train_loader, valid_loader, test_loader
