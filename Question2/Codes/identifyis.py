# -*- coding: utf-8 -*-
import torch.utils.data as data
import torch
import os
PAD_TOKEN = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
from torch.utils.data import DataLoader
class ias (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, id, unk='[UNK]'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['tokens'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, id.wtoid)
        self.slot_ids = self.mapping_seq(self.slots, id.stoid)
        self.intent_ids = self.mapping_lab(self.intents, id.itoid)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'tokens': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
