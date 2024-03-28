import unittest, json
from transformer import Transformer
from mha import MultiHeadAttention
from pe import PositionalEncoding

from gradescope_utils.autograder_utils.decorators import weight

import torch
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import json

import math
import time


filepaths = ['data/visible.de.code.json', 'data/visible.en.code.json']

device = 'cpu'

BATCH_SIZE = 10
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
LEN_DE_VOCAB = 19215
LEN_EN_VOCAB = 10838

EMB_DIM = 64
FFN_DIM = 128
ATTN_HEADS = 2
NUM_LAYERS = 2
MAX_SEQ_LEN = 200
DROPOUT_PROB = 0.1
MAX_INFERENCE_LENGTH = 50

def data_process(filepaths):
  with open(filepaths[0], 'r') as f:
    raw_de_iter = json.load(f)
  with open(filepaths[1], 'r') as f:
    raw_en_iter = json.load(f)
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor(raw_de,
                            dtype=torch.long)
    en_tensor_ = torch.tensor(raw_en,
                            dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data


def generate_batch(data_batch):
  de_batch, en_batch = [], []
  de_batch_lens, en_batch_lens = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch_lens.append(len(de_batch[-1]))
    en_batch_lens.append(len(en_batch[-1]))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX, batch_first=True)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)
  return de_batch, en_batch, torch.LongTensor(de_batch_lens), torch.LongTensor(en_batch_lens)

data = data_process(filepaths)


# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.decoder_inference_test_data = torch.load('data/visible_inference_output.pkl')

    @weight(8)
    def test_decoder_inference_outputs_extra_credit(self):
       
        import editdistance

        data_iter = DataLoader(data, batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=generate_batch)
        model = Transformer(src_vocab_size = LEN_DE_VOCAB,
                    tgt_vocab_size = LEN_EN_VOCAB, 
                    sos_idx = BOS_IDX, 
                    eos_idx = EOS_IDX,
                    d_model = EMB_DIM, 
                    num_heads = ATTN_HEADS, 
                    num_layers = NUM_LAYERS, 
                    d_ff = FFN_DIM, 
                    max_seq_length = MAX_SEQ_LEN, 
                    dropout_prob = DROPOUT_PROB).to(device)
        model.load_state_dict(torch.load('trained_de_en_state_dict.pt', map_location=torch.device(device)), strict=True)


        model.eval()
        candidate_corpus = []
        d_cnt = 0
        w_cnt = 0
        w_cnt_h = 0
        with torch.no_grad():

            for iterator_idx, (src, trg, src_lengths, trg_lengths) in enumerate(data_iter):
                src, trg, src_lengths, trg_lengths = src.to(device), trg.to(device), src_lengths.to(device), trg_lengths.to(device)
                output_list, decoder_cache = model.inference(src = src, src_lengths = src_lengths, max_output_length = MAX_INFERENCE_LENGTH)

                for output_item, trg_item, trg_length_item, src_item, src_length_item in zip(output_list, trg, trg_lengths, src, src_lengths):

                    ref = trg_item[:trg_length_item].tolist()[1:-1]
                    hyp = output_item.tolist()[1:-1]
                    candidate_corpus.append(hyp)
                
                    d_cnt += editdistance.eval(ref, hyp)
                    w_cnt += len(ref)
                    w_cnt_h += len(hyp)


        err_rate = d_cnt / w_cnt

        self.assertAlmostEqual(err_rate, self.decoder_inference_test_data["err_rate"], places=2, msg="Inference error rate significantly differs; check for major bugs")
        self.assertAlmostEqual(err_rate * 100, self.decoder_inference_test_data["err_rate"] * 100, places=1, msg="Inference error rate differs across the entire eval set")
        self.assertEqual(w_cnt_h, self.decoder_inference_test_data["hyp_len"], msg="Total inferenced length across all samples added differs")

        for item_idx in range(len(self.decoder_inference_test_data["inference_res"])):
            hyp = candidate_corpus[item_idx]
            ref_hyp = self.decoder_inference_test_data["inference_res"][item_idx]
            self.assertEqual(len(hyp), len(ref_hyp), msg=f"Inference length differs for the sample #{item_idx}")
            self.assertEqual(torch.sum(torch.LongTensor(hyp) != torch.LongTensor(ref_hyp)).item(), 0, msg=f"Inference result differs for the sample #{item_idx}")
           

       
    @weight(2)
    def test_decoder_inference_cache_extra_credit(self):
        data_iter = DataLoader(data, batch_size=BATCH_SIZE,
                       shuffle=False, collate_fn=generate_batch)
        model = Transformer(src_vocab_size = LEN_DE_VOCAB,
                    tgt_vocab_size = LEN_EN_VOCAB, 
                    sos_idx = BOS_IDX, 
                    eos_idx = EOS_IDX,
                    d_model = EMB_DIM, 
                    num_heads = ATTN_HEADS, 
                    num_layers = NUM_LAYERS, 
                    d_ff = FFN_DIM, 
                    max_seq_length = MAX_SEQ_LEN, 
                    dropout_prob = DROPOUT_PROB).to(device)
        model.load_state_dict(torch.load('trained_de_en_state_dict.pt', map_location=torch.device(device)), strict=True)


        model.eval()
        decoder_cache_item_list = []
        d_cnt = 0
        w_cnt = 0
        w_cnt_h = 0
        with torch.no_grad():

            for iterator_idx, (src, trg, src_lengths, trg_lengths) in enumerate(data_iter):
                src, trg, src_lengths, trg_lengths = src.to(device), trg.to(device), src_lengths.to(device), trg_lengths.to(device)
                output_list, decoder_cache = model.inference(src = src, src_lengths = src_lengths, max_output_length = MAX_INFERENCE_LENGTH)

                for decoder_cache_item_ in decoder_cache:
                   decoder_cache_item_list.append(decoder_cache_item_)
        for item_idx in range(len(self.decoder_inference_test_data["inference_decoder_cache"])):
            decoder_cache_item = decoder_cache_item_list[item_idx]
            ref_decoder_cache_item = self.decoder_inference_test_data["inference_decoder_cache"][item_idx]
            self.assertEqual(len(decoder_cache_item), NUM_LAYERS, msg=f"Decoder cache should be a list with length equal to the number of decoder layers #{item_idx}")
            for l_idx in range(len(ref_decoder_cache_item)):
                self.assertAlmostEqual(torch.sum(torch.abs(decoder_cache_item[l_idx] - ref_decoder_cache_item[l_idx])).item(), 0, places = 2, msg=f'Decoder cache during inference differs at layer #{l_idx}, for sample #{item_idx}')