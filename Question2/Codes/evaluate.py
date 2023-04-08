# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report

import torch
import conll
def evaluation_loop(data, criterion_slots, criterion_intents, model, id):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            intents, slots = model(sample['tokens'], id)
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [id.idtoi[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [id.idtoi[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['tokens'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [id.idtos[elem] for elem in gt_ids[:length]]
                utterance = [id.idtow[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], id.idtos[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = conll.evaluate(ref_slots, hyp_slots)
        print("results",results)
    
    except Exception as ex:
    #    # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
     
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array