from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import shutil

import torch.utils.data as data
from PIL import Image
import PIL
import os.path
import pickle
import random
import h5py
import numpy as np
import pandas as pd
import pprint

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import re
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Loading BERT embedder and tokenizer
        self.embedder = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)  
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased') 
        print("bert")

    
    def embd(self, tokens_id):
        #Generate word level and sentence level embedding tokens from tokens_id
        tokens_id_tensor = tokens_id.unsqueeze(0) 
        outputs = self.embedder(tokens_id_tensor)

        sentence = outputs[1]
        sentence = sentence.squeeze(0)

        words = outputs[0]
        words = words.squeeze(0)
        wordsm = words.t()
        return sentence, wordsm

    def encode(self,input):
        input = input.unsqueeze(0)
        output = self.embedder(input)
        return output

    def tensor_tokens_id(self,inputs):
        tokens = self.tokenizer.tokenize(inputs)
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens_id

    def id_tensor_tokens(self,tokens_id):
        tokens = self.tokenizer.convert_ids_to_tokens(tokens_id)
        return tokens

    def forward(self, captions):
        i = 0
        sent_emb = np.zeros((30, 768), dtype='float32')
        sent_emb = torch.as_tensor(torch.from_numpy(sent_emb), dtype=torch.float32)
        words_emb = np.zeros((30, 768, 18), dtype='float32')
        words_emb = torch.as_tensor(torch.from_numpy(words_emb), dtype=torch.float32)
        sent_emb = sent_emb.cuda()
        words_emb = words_emb.cuda()
        while (i < 30):
            captions1 = captions[i].cpu()
            captions1 = captions1.numpy().tolist()
            captions1 = torch.LongTensor(captions1)
            captions1 = captions1.cuda()
            sent_emb[i], words_emb[i] = self.embd(captions1)
            print(sent_emb, words_emb)
            i = i + 1
        return sent_emb,words_emb