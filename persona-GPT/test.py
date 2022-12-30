#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:10:21 2020

@author: af1tang
"""
import torch, os, pickle, time
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from load_configs import tokenizer, pretrain_stats, train_stats, opts, create_dir, p1_tok, p2_tok, start_tok, act_tok
from utils import *
from accelerate import Accelerator
accelerator = Accelerator()
device=accelerator.device
print(device)

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelWithLMHead


def evaluate_loop(data):
    model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium').to(device)
    dataloader = DataLoader(data, batch_size=opts.eval_batch_size, shuffle=True); del data
    data_iter = iter(dataloader)
    with torch.no_grad():
        eval_stats, total_steps, val_loss, val_f1_score = {}, 0, 0.0, 0.0
        model.eval()
        for i in range(len(dataloader)):
            batch = next(data_iter)
            xx,yy = batch
            print(xx)
            print(len(xx))
            print(yy)
            print(len(yy))
            try:
                xx, yy = torch.stack(xx, -1).to(device), torch.stack(yy, -1).to(device)
            except:
                xx, yy = to_var(xx), to_var(yy)

            ## forward on new data batch
            _outp = model(xx)
            past = _outp.past_key_values
            outp = model(yy, past_key_values=past, labels=yy)
            loss = outp[0]
            ytrue=np.array( filter_turn_indices(to_data(yy[...,1:].contiguous().view(-1)) ) )
            ypred=np.array( filter_turn_indices(to_data( outp[1][..., :-1, :].contiguous().topk(1)[1].view(-1)) ) ) 
            min_len = min(len(ypred), len(ytrue))
            hits = [set(ypred[i]).intersection(set(ytrue[i])) for i in range(min_len)]
            prec = [len(hits[i])/len(ypred[i]) for i in range(min_len)]
            rec = [len(hits[i])/len(ytrue[i]) for i in range(min_len)]
            f1 = np.mean([2*(prec[i]*rec[i])/(prec[i] + rec[i]+1e-3) for i in range(min_len)])
            val_f1_score += f1
            val_loss += loss.mean().item()
            total_steps +=1 
            #if total_steps%100 ==0: print("... %d out of %d"%(total_steps, len(dataloader)))
            
    val_loss = val_loss / total_steps 
    val_f1_score = val_f1_score / total_steps
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    eval_stats = {'perplexity': perplexity, 'loss': val_loss, 'f1': val_f1_score}
    print("Done.")
    return eval_stats

if __name__ == '__main__':        

    print("="*50)
    print("Evaluating ... ")
    with open('/home/isds1/persona/personaGPT/model_check-batch-train-epoch20/valid_data', 'rb') as f: eval_data = pickle.load(f)
    eval_stats = evaluate_loop(eval_data)
    print("Done!")
    print()
    print("Perplexity: %.2f" %eval_stats['perplexity'])
    print("F1 Score: %.2f" % eval_stats['f1'])
    print("="*50)
