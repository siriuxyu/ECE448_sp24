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
        self.mha_test_data = torch.load('data/visible_mha.pkl')
        self.pe_test_data = torch.load('data/visible_pe.pkl')
        self.encoder_test_data = torch.load('data/visible_encoder_states.pkl')
        self.decoder_test_data = torch.load('data/visible_decoder_output.pkl')

    @weight(15)
    def test_mha_no_mask(self):
        torch.manual_seed(23631)

        mha_random = MultiHeadAttention(d_model=64, num_heads=2)

        mha_random.load_state_dict(self.mha_test_data["mha_random"])
        qe_random = self.mha_test_data["qe_random"]


        ## TEST ONE; NO MASK
        q, k, v = mha_random.compute_mh_qkv_transformation(Q = qe_random, K=qe_random, V = qe_random)
        attn = mha_random.compute_scaled_dot_product_attention(q, k, v, key_padding_mask = None, attention_mask = None)

        ref_q = self.mha_test_data["q_1"]
        ref_k = self.mha_test_data["k_1"]
        ref_v = self.mha_test_data["v_1"]
        ref_attn = self.mha_test_data["attn_1"]
        
        for ts, ref_ts in zip([q, k, v], [ref_q, ref_k, ref_v]):
            self.assertEqual(ts.shape[0], ref_ts.shape[0],
                            'dim 0 mismatch in transformed q/k/v')
            self.assertEqual(ts.shape[1], ref_ts.shape[1],
                            'dim 1 mismatch in transformed q/k/v')
            self.assertEqual(ts.shape[2], ref_ts.shape[2],
                            'dim 2 mismatch in transformed q/k/v')
            self.assertEqual(ts.shape[3], ref_ts.shape[3],
                             'dim 3 mismatch in transformed q/k/v')
            
            self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='q/k/v transformation incorrect. Check your compute_mh_qkv_transformation')

        ts = attn
        ref_ts = ref_attn

        self.assertEqual(ts.shape[0], ref_ts.shape[0],
                        'dim 0 mismatch in compute_scaled_dot_product_attention ret val')
        self.assertEqual(ts.shape[1], ref_ts.shape[1],
                        'dim 1 mismatch in compute_scaled_dot_product_attention ret val')
        self.assertEqual(ts.shape[2], ref_ts.shape[2],
                        'dim 2 mismatch in compute_scaled_dot_product_attention ret val')
        
        self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='Attention computation incorrect. Check your compute_mh_qkv_transformation')

    @weight(5)
    def test_mha_key_padding_mask(self):
        torch.manual_seed(23631)

        mha_random = MultiHeadAttention(d_model=64, num_heads=2)

        mha_random.load_state_dict(self.mha_test_data["mha_random"])
        q_random = self.mha_test_data["q_random"]

        ## TEST TWO; key_padding_mask 
        key_padding_mask = self.mha_test_data["key_padding_mask_2"]
        q, k, v = mha_random.compute_mh_qkv_transformation(Q = q_random, K=q_random, V = q_random)
        attn = mha_random.compute_scaled_dot_product_attention(q, k, v, key_padding_mask = key_padding_mask, attention_mask = None)

        ref_attn = self.mha_test_data["attn_2"]

        ts = attn
        ref_ts = ref_attn

        self.assertEqual(ts.shape[0], ref_ts.shape[0],
                        'dim 0 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask')
        self.assertEqual(ts.shape[1], ref_ts.shape[1],
                        'dim 1 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask')
        self.assertEqual(ts.shape[2], ref_ts.shape[2],
                        'dim 2 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask')
        
        self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='Attention computation incorrect w/ key_padding_mask. Check your compute_mh_qkv_transformation')
    
    @weight(5)
    def test_mha_key_padding_mask_attention_mask(self):
        torch.manual_seed(23631)

        mha_random = MultiHeadAttention(d_model=64, num_heads=2)

        mha_random.load_state_dict(self.mha_test_data["mha_random"])
        v_random = self.mha_test_data["v_random"]

        ## TEST THREE; key_padding_mask and attention_mask
        key_padding_mask = self.mha_test_data["key_padding_mask_3"]
        attention_mask = self.mha_test_data["attention_mask_3"]
        q, k, v = mha_random.compute_mh_qkv_transformation(Q = v_random, K=v_random, V = v_random)
        attn = mha_random.compute_scaled_dot_product_attention(q, k, v, key_padding_mask = key_padding_mask, attention_mask = attention_mask)

        ref_attn = self.mha_test_data["attn_3"]
        
        ts = attn
        ref_ts = ref_attn

        self.assertEqual(ts.shape[0], ref_ts.shape[0],
                        'dim 0 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask and attention_mask')
        self.assertEqual(ts.shape[1], ref_ts.shape[1],
                        'dim 1 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask and attention_mask')
        self.assertEqual(ts.shape[2], ref_ts.shape[2],
                        'dim 2 mismatch in compute_scaled_dot_product_attention ret val w/ key_padding_mask and attention_mask')
        
        self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='Attention computation incorrect w/ key_padding_mask and attention_mask. Check your compute_mh_qkv_transformation')

    @weight(5)
    def test_mha_different_query_and_key(self):
        torch.manual_seed(23631)

        mha_random = MultiHeadAttention(d_model=64, num_heads=2)

        mha_random.load_state_dict(self.mha_test_data["mha_random"])
        q_random = self.mha_test_data["q_random"]
        v_random = self.mha_test_data["v_random"]
        ## TEST FOUR; different query and k/v length
        key_padding_mask =  self.mha_test_data["key_padding_mask_4"]
        q, k, v = mha_random.compute_mh_qkv_transformation(Q = q_random, K=v_random, V = v_random)
        attn = mha_random.compute_scaled_dot_product_attention(q, k, v, key_padding_mask = key_padding_mask, attention_mask = None)
        
        ref_q = self.mha_test_data["q_4"]
        ref_k = self.mha_test_data["k_4"]
        ref_v = self.mha_test_data["v_4"]
        ref_attn = self.mha_test_data["attn_4"]
        
        for ts, ref_ts in zip([q, k, v], [ref_q, ref_k, ref_v]):
            self.assertEqual(ts.shape[0], ref_ts.shape[0],
                            'dim 0 mismatch in transformed q/k/v when query and key/value have different time steps')
            self.assertEqual(ts.shape[1], ref_ts.shape[1],
                            'dim 1 mismatch in transformed q/k/v when query and key/value have different time steps')
            self.assertEqual(ts.shape[2], ref_ts.shape[2],
                            'dim 2 mismatch in transformed q/k/v when query and key/value have different time steps')
            self.assertEqual(ts.shape[3], ref_ts.shape[3],
                             'dim 3 mismatch in transformed q/k/v when query and key/value have different time steps')
            
            self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='q/k/v transformation incorrect when query and key/value have different time steps. Check your compute_mh_qkv_transformation')

        ts = attn
        ref_ts = ref_attn

        self.assertEqual(ts.shape[0], ref_ts.shape[0],
                        'dim 0 mismatch in compute_scaled_dot_product_attention ret val when query and key/value have different time steps')
        self.assertEqual(ts.shape[1], ref_ts.shape[1],
                        'dim 1 mismatch in compute_scaled_dot_product_attention ret val when query and key/value have different time steps')
        self.assertEqual(ts.shape[2], ref_ts.shape[2],
                        'dim 2 mismatch in compute_scaled_dot_product_attention ret val when query and key/value have different time steps')
        
        self.assertAlmostEqual(torch.sum(torch.abs(ts - ref_ts)).item(), 0, places = 2, msg='Attention computation incorrect when query and key/value have different time steps. Check your compute_mh_qkv_transformation')


    @weight(5)
    def test_pe(self):
        torch.manual_seed(23631)

        pe = PositionalEncoding(d_model=self.pe_test_data["d_model"], max_seq_length=self.pe_test_data["max_seq_length"])

        x_in = self.pe_test_data["x"]
        x_out = pe(x_in)

        self.assertAlmostEqual(torch.sum(torch.abs(pe.pe - self.pe_test_data["pe"])).item(), 0, places = 2, msg='Positional Encoding has incorrect encoding entries')

        self.assertEqual(x_out.shape[0], self.pe_test_data["pe_output"].shape[0],
                        'dim 0 mismatch in forward function of PositionalEncoding')
        self.assertEqual(x_out.shape[1], self.pe_test_data["pe_output"].shape[1],
                        'dim 1 mismatch in forward function of PositionalEncoding')
        self.assertEqual(x_out.shape[2], self.pe_test_data["pe_output"].shape[2],
                        'dim 2 mismatch in forward function of PositionalEncoding')

        self.assertAlmostEqual(torch.sum(torch.abs(x_out - self.pe_test_data["pe_output"])).item(), 0, places = 2, msg='Positional Encoding has incorrect output')

    @weight(30)
    def test_encoder_output(self):
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

        ref_batched_output_encoder = self.encoder_test_data

        with torch.no_grad():

            for it_idx, (src, trg, src_lengths, trg_lengths) in enumerate(data_iter):
                src, trg, src_lengths, trg_lengths = src.to(device), trg.to(device), src_lengths.to(device), trg_lengths.to(device)

                output_encoder, _ = model.forward_encoder(src = src, src_lengths = src_lengths)

                cur_ref_batch_output_encoder = ref_batched_output_encoder[it_idx]

                for item_idx in range(cur_ref_batch_output_encoder.size(0)):
                    ref_enc_item = cur_ref_batch_output_encoder[item_idx, :src_lengths[item_idx]]
                    enc_item = output_encoder[item_idx, :src_lengths[item_idx]]
                    self.assertAlmostEqual(torch.sum(torch.abs(ref_enc_item - enc_item)).item(), 0, places = 2, msg=f'The encoder output for the sample #{item_idx} in the batch #{it_idx} is not correct')


    @weight(30)
    def test_encoder_decoder_predictions(self):
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

        sum_tokens = 0
        sum_tokens_accuracy = 0

        all_batched_output = []
        all_batched_output_preds = []

        with torch.no_grad():

            for it_idx, (src, trg, src_lengths, trg_lengths) in enumerate(data_iter):
                src, trg, src_lengths, trg_lengths = src.to(device), trg.to(device), src_lengths.to(device), trg_lengths.to(device)
                
                output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_lengths = trg_lengths)

                output = output[:, :-1, :].contiguous()
                trg = trg[:, 1:].contiguous()

                all_batched_output.append(output)


                output_pred = output[trg != PAD_IDX].argmax(-1)
                trg = trg[trg != PAD_IDX]

                all_batched_output_preds.append(output_pred)


                sum_tokens_accuracy += torch.sum(output_pred == trg).item()
                sum_tokens += len(trg)


        token_accuracy = sum_tokens_accuracy / sum_tokens
        self.assertAlmostEqual(token_accuracy, self.decoder_test_data["decoder_acc"], places=2, msg="Decoder accuracy significantly differs; check for major bugs")
        self.assertAlmostEqual(token_accuracy * 100, self.decoder_test_data["decoder_acc"] * 100, places=1, msg="Decoder accuracy differs across the entire eval set")

        for batch_idx in range(len(self.decoder_test_data["decoder_preds"])):
            pred = all_batched_output_preds[batch_idx]
            ref_pred = self.decoder_test_data["decoder_preds"][batch_idx]
            self.assertEqual(torch.sum(pred != ref_pred).item(), 0, msg=f"Decoder prediction has discrepancies in batch #{batch_idx}")


    @weight(5)
    def test_encoder_decoder_states(self):
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

        sum_tokens = 0
        sum_tokens_accuracy = 0

        with torch.no_grad():

            for it_idx, (src, trg, src_lengths, trg_lengths) in enumerate(data_iter):
                src, trg, src_lengths, trg_lengths = src.to(device), trg.to(device), src_lengths.to(device), trg_lengths.to(device)
                
                output = model(src = src, tgt = trg, src_lengths = src_lengths, tgt_lengths = trg_lengths)

                output = output[:, :-1, :].contiguous()
                trg = trg[:, 1:].contiguous()

                ref_output = self.decoder_test_data["decoder_states"][it_idx]

                for item_idx in range(ref_output.size(0)):
                    ref_dec_item = ref_output[item_idx,:trg_lengths[item_idx]]
                    dec_item = output[item_idx,:trg_lengths[item_idx]]
                    self.assertAlmostEqual(torch.sum(torch.abs(ref_dec_item - dec_item)).item(), 0, places=2, msg=f"The decoder output for the sample #{item_idx} in the batch #{it_idx} is not correct")
