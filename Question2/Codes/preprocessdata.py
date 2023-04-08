import pandas as pd
import json
PAD_TOKEN = 0
from identifyis import ias
from torch.utils.data import DataLoader
import torch

import os
PAD_TOKEN = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'	
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['tokens']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['tokens'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our seleceted device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["tokens"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

def wtoid_func(raw_dataset):
# returns a dictionary of words and their ids
    words = []
    for entry in raw_dataset:
       words.extend(entry['tokens'].split())
    words = list(set(words))
    words_dict = {'[PAD]': PAD_TOKEN}
    words_dict.update({w:i+1 for i, w in enumerate(words)})
    words_dict['[UNK]'] = len(words_dict)
    return words_dict

def stoid_func(raw_dataset):
# returns a dictionary of slots and their ids
    slots = ['[PAD]']
    for entry in raw_dataset:
       slots.extend(entry['slots'].split())
    slots = list(set(slots))
    slots_dict = {s:i for i, s in enumerate(slots)}
    return slots_dict

def itoid_func(raw_dataset):
# returns a dictionary of intents and their ids
    intents = [entry['intent'] for entry in raw_dataset]
    intents = list(set(intents))
    intents_dict = {inte:i for i, inte in enumerate(intents)}
    return intents_dict

def vocab_func(raw_dataset):
    vocab = set()
    for entry in raw_dataset:
        vocab = vocab.union(set(entry['tokens'].split()))
    return ['[PAD]'] + list(vocab) + ['[UNK]']

class idcreation():
    def __init__(self, training_set, validation_set, testing_set):
        self.wtoid = wtoid_func(training_set + validation_set + testing_set)
        self.stoid = stoid_func(training_set + validation_set + testing_set)
        self.itoid = itoid_func(training_set + validation_set + testing_set)
        self.vocab = vocab_func(training_set + validation_set + testing_set)
        self.idtow = {v:k for k, v in self.wtoid.items()}
        self.idtos = {v:k for k, v in self.stoid.items()}
        self.idtoi = {v:k for k, v in self.itoid.items()}


def access(path):
    
    df=pd.read_csv(path)
    #test=pd.read_csv(path+"atis.test"+".csv")
    #dev=pd.read_csv(path+"atis.dev"+".csv")
    read_obj=df.to_json(orient='records')
    dataset = json.loads(read_obj)
    return dataset
    


def process(path):
    training_set = access(path+"atis.train"+".csv")
    testing_set = access(path+"atis.test"+".csv")
    validation_set = access(path+"atis.dev"+".csv")
    
    
    id = idcreation(training_set, validation_set, testing_set)

    ##############################
    training_data = ias(training_set, id)
    validation_data = ias(validation_set, id)
    testing_data = ias(testing_set, id)
    
    ##############################
    training_load = DataLoader(training_data, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    validation_load = DataLoader(validation_data, batch_size=64, collate_fn=collate_fn)
    testing_load = DataLoader(testing_data, batch_size=64, collate_fn=collate_fn)
    
    return training_load, validation_load, testing_load, id
'''
def main(path):
    training_load, validation_load, testing_load, id=process(path)
    hL = 10
    emb = 10

    lr = 0.0001 # learning rate
    clip = 5 # clip clipping	

    sidout = len(id.stoid)
    iidout = len(id.itoid)
    vocab_len = len(id.wtoid)

    training_set = access(path+"atis.train"+".csv")
    testing_set = access(path+"atis.test"+".csv")
    validation_set = access(path+"atis.dev"+".csv")
'''         



