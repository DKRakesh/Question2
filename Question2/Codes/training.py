# -*- coding: utf-8 -*-
from bertmodule import bert
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import preprocessdata as pre
import numpy as np
import evaluate
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'





def training_loop(data, optimizer, criterion_slots, criterion_intents, model, id, clip):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the clip
        intent, slots = model(sample['tokens'], id)
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def main(path):
    	
    lr = 10 # learning rate
    clip = 5 # clip clipping

    #path = "C:/Users/LENOVO/nlp/Question2/Dataset/"
    training_load, validation_load, testing_load, id=pre.process(path)
    training_set = pre.access(path+"atis.train"+".csv")
    testing_set = pre.access(path+"atis.test"+".csv")
    validation_set = pre.access(path+"atis.dev"+".csv")
    id = pre.idcreation(training_set, validation_set, testing_set)
    sidout = len(id.stoid)
    iidout = len(id.itoid)
    vocab_len = len(id.wtoid)

    model = bert(iidout, sidout, id)
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss()
    criterion_intents = nn.CrossEntropyLoss(ignore_index=0)
    
    n_epochs = 2
    patience = 1
    
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    from tqdm import tqdm
    for x in tqdm(range(1,n_epochs)):
        loss = training_loop(training_load, optimizer, criterion_slots, 
                        criterion_intents, model, id, clip)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = evaluate.evaluation_loop(validation_load, criterion_slots, 
                                                        criterion_intents, model, id)
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev['total']['f']
            
            if f1 > best_f1:
                best_f1 = f1
            else:
                if patience%3 == 0:
                    # halve optimizer learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean
    results_test, intent_test, _ = evaluate.evaluation_loop(testing_load, criterion_slots, 
                                            criterion_intents, model, id)


#print (f"#################### bert - {dataset} ####################")
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])