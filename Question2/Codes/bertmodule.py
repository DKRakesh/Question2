from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import os
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
class bert(nn.Module):
    def __init__ (self, out_int, out_slot, lang):
        super(bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(lang.vocab)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(device)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        # last hidden size of BERT = 768
        self.intent_classifier = nn.Linear(768, out_int)
        self.slot_classifier = nn.Linear(768, out_slot)
        
    def forward(self, input, lang):
        # get back the input sentence
        utterance = []
        for element in input:
            utterance.append(' '.join(lang.vocab[i] for i in element if i > 0))
        tokenized = self.tokenizer(utterance, return_tensors='pt', add_special_tokens=True,padding=True, truncation=True).to(device)
        bert_out = self.bert(**tokenized)
        # get intent output from pooled output
        intent = bert_out.pooler_output
        # get slot output from last hidden state
        slots = bert_out.last_hidden_state[:,:input.size(1),:]
        
        intent = self.intent_classifier(intent)
        slots = self.slot_classifier(slots)
        slots = slots.permute(0, 2, 1)
        return intent, slots# -*- coding: utf-8 -*-

