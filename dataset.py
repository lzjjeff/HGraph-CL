
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from util import config
import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


class MoseiDataset(Dataset):
    def __init__(self, data_dic, dname="train", embed_type="bert_word", aligned=True, v_max=1.0, a_max=1.0, n_samples=None):
        self.data_dic = data_dic[dname]
        self.fold_keys = list(data_dic[dname].keys())
        self.embed_type = embed_type
        self.aligned = aligned
        if n_samples:
            self.fold_keys = self.fold_keys[:n_samples]
        self.v_max = v_max
        self.a_max = a_max

    def __getitem__(self, idx):
        key = self.fold_keys[idx]

        acoustic = self.data_dic[key]['a']
        acoustic[~np.isfinite(acoustic)] = 0
        acoustic_normed = acoustic / self.a_max

        visual = self.data_dic[key]['v']
        visual[~np.isfinite(visual)] = 0
        visual_normed = visual / self.v_max

        if not self.aligned:    # utterance-level
            a_len = self.data_dic[key]['a_len']
            v_len = self.data_dic[key]['v_len']

            data = {"id": key, "text": self.data_dic[key]['t'], "visual": visual,
                    "acoustic": acoustic, "visual_normed": visual_normed,
                    "acoustic_normed": acoustic_normed, "label": self.data_dic[key]['l'][0],
                    "visual_length": v_len, "acoustic_length": a_len}
        else:
            data = {"id": key, "text": self.data_dic[key]['t'], "visual": visual,
                    "acoustic": acoustic, "visual_normed": visual_normed,
                    "acoustic_normed": acoustic_normed, "label": self.data_dic[key]['l'][0]}

        if self.embed_type == "glove":
            data["words"] = self.data_dic[key]['words']

        return data

    def __len__(self):
        return len(self.fold_keys)


def sort_sequences(inputs, lengths):
    """sort_sequences
    Sort sequences according to lengths descendingly.

    :param inputs (Tensor): input sequences, size [B, T, D]
    :param lengths (Tensor): length of each sequence, size [B]
    """
    lengths = torch.Tensor(lengths)
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()
    return inputs[sorted_idx], lengths_sorted, sorted_idx


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix_t = np.zeros((len(words), len(words))).astype('float32')
    matrix_va = np.triu(np.tril(np.ones((len(words), len(words))).astype('float32'), 1), -1)
    matrix_inter = np.ones((len(words), len(words))).astype('float32')  # inter_modality
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix_t[token.i][token.i] = 1
        for child in token.children:
            matrix_t[token.i][child.i] = 1
            matrix_t[child.i][token.i] = 1

    return matrix_t, matrix_va,  matrix_inter


def dependency_adj_matrix_4_bert(text, re_idx):
    tokens = nlp(text)
    words = text.split()
    matrix_t = np.zeros((len(re_idx), len(re_idx))).astype('float32')   # intra_modality
    matrix_va = np.ones((len(re_idx), len(re_idx))).astype('float32')   # inter_modality
    matrix_inter = np.ones((len(re_idx), len(re_idx))).astype('float32')   # inter_modality
    assert len(words) == len(list(tokens)) == re_idx[-1].item()+1

    for token in tokens:
        t_idx = torch.where(re_idx==token.i)[0].numpy().tolist()
        matrix_t[t_idx[0]:t_idx[-1]+1, t_idx[0]:t_idx[-1]+1] = 1
        for child in token.children:
            c_idx = torch.where(re_idx==child.i)[0].numpy().tolist()
            matrix_t[t_idx[0]:t_idx[-1]+1, c_idx[0]:c_idx[-1]+1] = 1
            matrix_t[c_idx[0]:c_idx[-1]+1, t_idx[0]:t_idx[-1]+1] = 1

    return matrix_t, matrix_va,  matrix_inter


def collate_fn(batch):
    MAX_LEN = config['max_len']

    if config["embed_type"] == "glove":
        for utt in batch:
            # get adj
            adj_matrix_t, adj_matrix_va, adj_matrix_inter = dependency_adj_matrix(' '.join(utt["words"]))
            utt["adj_matrix_t"] = adj_matrix_t
            utt["adj_matrix_va"] = adj_matrix_va
            utt["adj_matrix_inter"] = adj_matrix_inter

        lens = [min(len(row["text"]), MAX_LEN) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)

        tdims = batch[0]["text"].shape[1]
        adims = batch[0]["acoustic"].shape[1]
        vdims = batch[0]["visual"].shape[1]

        text_tensor = torch.zeros((bsz, max_seq_len, tdims))
        visual_tensor = torch.zeros((bsz, max_seq_len, vdims))
        acoustic_tensor = torch.zeros((bsz, max_seq_len, adims))
        visual_normed_tensor = torch.zeros((bsz, max_seq_len, vdims))
        acoustic_normed_tensor = torch.zeros((bsz, max_seq_len, adims))
        adj_matrix = torch.zeros((bsz, 3 * max_seq_len, 3 * max_seq_len))

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            text_tensor[i_batch, :length] = torch.Tensor(input_row["text"][:length])
            visual_tensor[i_batch, :length] = torch.Tensor(input_row["visual"][:length])
            acoustic_tensor[i_batch, :length] = torch.Tensor(input_row["acoustic"][:length])
            visual_normed_tensor[i_batch, :length] = torch.Tensor(input_row["visual_normed"][:length])
            acoustic_normed_tensor[i_batch, :length] = torch.Tensor(input_row["acoustic_normed"][:length])

            adj_mat_t = np.zeros((max_seq_len, max_seq_len))
            adj_mat_va = np.zeros((max_seq_len, max_seq_len))
            adj_mat_inter = np.zeros((max_seq_len, max_seq_len))
            adj_mat_t[:length, :length] = input_row["adj_matrix_t"][:length, :length]
            adj_mat_va[:length, :length] = input_row["adj_matrix_va"][:length, :length]
            adj_mat_inter[:length, :length] = input_row["adj_matrix_inter"][:length, :length]
            adj_matrix[i_batch] = torch.FloatTensor(np.block([[adj_mat_t, adj_mat_inter, adj_mat_inter],
                                                              [adj_mat_inter, adj_mat_va, adj_mat_inter],
                                                              [adj_mat_inter, adj_mat_inter, adj_mat_va]]))

        tgt_tensor = torch.stack([torch.tensor(row["label"]) for row in batch])
        text_tensor, lens, sorted_idx = sort_sequences(text_tensor, lens)

        return text_tensor, visual_tensor[sorted_idx], acoustic_tensor[sorted_idx], \
               visual_normed_tensor[sorted_idx], acoustic_normed_tensor[sorted_idx], \
               adj_matrix[sorted_idx], tgt_tensor[sorted_idx], lens
    else:
        for utt in batch:
            text = " ".join(utt["text"])
            inputs = tokenizer(text, return_tensors="pt")

            re_idx = []

            for idx, word in enumerate(utt["text"]):
                if idx != 0: word = " " + word  # for roberta
                word_inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
                word_input_ids = word_inputs["input_ids"]
                wlen = word_input_ids.size(1)
                re_idx.extend([idx] * wlen)

            assert len(re_idx) == (inputs["input_ids"].size(1) - 2)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            utt["text"] = input_ids.squeeze(0)
            utt["attention_mask"] = attention_mask.squeeze(0)

            re_idx = torch.LongTensor(re_idx)

            # get adj
            adj_matrix_t, adj_matrix_va, adj_matrix_inter = dependency_adj_matrix_4_bert(text, re_idx)
            utt["adj_matrix_t"] = adj_matrix_t
            utt["adj_matrix_va"] = adj_matrix_va
            utt["adj_matrix_inter"] = adj_matrix_inter

            acoustic_tensor = torch.from_numpy(utt["acoustic"])
            utt["acoustic"] = torch.index_select(acoustic_tensor, 0, re_idx).numpy()

            visual_tensor = torch.from_numpy(utt["visual"])
            utt["visual"] = torch.index_select(visual_tensor, 0, re_idx).numpy()

            acoustic_normed_tensor = torch.from_numpy(utt["acoustic_normed"])
            utt["acoustic_normed"] = torch.index_select(acoustic_normed_tensor, 0, re_idx).numpy()

            visual_normed_tensor = torch.from_numpy(utt["visual_normed"])
            utt["visual_normed"] = torch.index_select(visual_normed_tensor, 0, re_idx).numpy()

            assert len(utt["visual"]) == len(utt["acoustic"]) == len(utt["visual_normed"]) \
                   == len(utt["acoustic_normed"]) == len(utt["text"]) - 2

        lens = [min(len(row["text"]) - 2, MAX_LEN) for row in batch]
        bsz, max_seq_len = len(batch), max(lens)

        adims = batch[0]["acoustic"].shape[1]
        vdims = batch[0]["visual"].shape[1]

        text_tensor = torch.zeros((bsz, max_seq_len + 2), dtype=torch.long)
        visual_tensor = torch.zeros((bsz, max_seq_len, vdims))
        acoustic_tensor = torch.zeros((bsz, max_seq_len, adims))
        visual_normed_tensor = torch.zeros((bsz, max_seq_len, vdims))
        acoustic_normed_tensor = torch.zeros((bsz, max_seq_len, adims))
        token_type_ids_tensor = torch.zeros((bsz, max_seq_len + 2), dtype=torch.long)
        attention_mask_tensor = torch.zeros((bsz, max_seq_len + 2), dtype=torch.long)
        adj_matrix = torch.zeros((bsz, 3 * max_seq_len, 3 * max_seq_len))

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            text_tensor[i_batch, :length + 2] = torch.LongTensor(input_row["text"][:length + 2])
            visual_tensor[i_batch, :length] = torch.Tensor(input_row["visual"][:length])
            acoustic_tensor[i_batch, :length] = torch.Tensor(input_row["acoustic"][:length])
            visual_normed_tensor[i_batch, :length] = torch.Tensor(input_row["visual_normed"][:length])
            acoustic_normed_tensor[i_batch, :length] = torch.Tensor(input_row["acoustic_normed"][:length])
            attention_mask_tensor[i_batch, :length + 2] = torch.LongTensor(input_row["attention_mask"][:length + 2])

            adj_mat_t = np.zeros((max_seq_len, max_seq_len))
            adj_mat_va = np.zeros((max_seq_len, max_seq_len))
            adj_mat_inter = np.zeros((max_seq_len, max_seq_len))
            adj_mat_t[:length, :length] = input_row["adj_matrix_t"][:length, :length]
            adj_mat_va[:length, :length] = input_row["adj_matrix_va"][:length, :length]
            adj_mat_inter[:length, :length] = input_row["adj_matrix_inter"][:length, :length]
            adj_matrix[i_batch] = torch.FloatTensor(np.block([[adj_mat_t, adj_mat_inter, adj_mat_inter],
                                                              [adj_mat_inter, adj_mat_va, adj_mat_inter],
                                                              [adj_mat_inter, adj_mat_inter, adj_mat_va]]))

        tgt_tensor = torch.stack([torch.tensor(row["label"]) for row in batch])
        text_tensor, lens, sorted_idx = sort_sequences(text_tensor, lens)

        return text_tensor, visual_tensor[sorted_idx], acoustic_tensor[sorted_idx], \
               visual_normed_tensor[sorted_idx], acoustic_normed_tensor[sorted_idx], \
               token_type_ids_tensor[sorted_idx], attention_mask_tensor[sorted_idx], \
               adj_matrix[sorted_idx], tgt_tensor[sorted_idx], lens
