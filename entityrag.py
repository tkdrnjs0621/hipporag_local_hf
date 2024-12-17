
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import re
from functools import partial
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import json
import copy
import igraph as ig

import pandas as pd
import numpy as np

def rank_docs(row, top_k):
    num_query_entities = len(row['nearest_entities'])
    ranked_docs = []
    searching_idx = [0]*num_query_entities
    
    large_idx=0
    # while(len(ranked_docs)<len(list_docs)):
    while(len(ranked_docs)<top_k):
        entity_idx = large_idx%num_query_entities
        # print(row['nearest_entities'][entity_idx][searching_idx[entity_idx]])
        # print(searching_idx[entity_idx],entity_idx, num_query_entities, len(ranked_docs))
        searching_doc = entity_to_doc[row['nearest_entities'][entity_idx][searching_idx[entity_idx]]]
        # print(searching_doc)
        # print(list(searching_doc)[0], list_docs[list(searching_doc)[0]])
        # print(row['nearest_entities'][entity_idx][searching_idx[entity_idx]],list_entities[row['nearest_entities'][entity_idx][searching_idx[entity_idx]]])
        # assert()
        ifnotadded=True
        for x in searching_doc:
            if(x not in ranked_docs):  
                # print(entity_idx, searching_idx[entity_idx],list_docs[x],list_entities[row['nearest_entities'][entity_idx][searching_idx[entity_idx]]],row['nearest_entities'][entity_idx][searching_idx[entity_idx]],len(searching_doc))              
                ranked_docs.append(x)
                large_idx+=1
                ifnotadded=False
                break
        if(ifnotadded):
            searching_idx[entity_idx]+=1
    # assert()
    row['ranked_idx'] = ranked_docs[:top_k]
    return row

def textify_result(row,list_docs,list_entities):
    row['retrieved_titles']= [list_docs[idx] for idx in row['ranked_idx']]
    row['nearest_entities_name']=[list_entities[e[0]] for e in row['nearest_entities']]
    return row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_query_path', type=str, default='data/entityrag/musique_query_ner_relevant.jsonl')
    parser.add_argument("--dataset_t2p_path", type=str, default="data/musique_dev_t2p.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_document_entity_map_path", type=str, default="data/entityrag/musique_doc_ent_map.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_entity_path", type=str, default="data/entityrag/musique_entity_list.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="data/entityrag/musique_result.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument('--topk_max', type=int, default=20, help="A list of numbers")
    parser.add_argument('--option',type=str, choices=['v0, v1'], default='v0')

    args = parser.parse_args()
    
    query = load_dataset('json', data_files=args.dataset_query_path)["train"]
    t2p = load_dataset('json', data_files=args.dataset_t2p_path)["train"]
    list_docs = [row['title'] for row in t2p]

    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    with open(args.dataset_document_entity_map_path, 'r') as f:
        list_demap=json.loads(f.readline())

    entity_to_doc = {}
    for i in list_demap:
        entity_to_doc[int(i[1])] = entity_to_doc.get(int(i[1]), set()) | {int(i[0])}
        
    if(args.option=='v0'):
        result = query.map(partial(rank_docs, top_k=args.topk_max))
    elif(args.option=='v1'):
        assert()
        
    result = result.map(partial(textify_result, list_docs=list_docs, list_entities=list_entities))
    result.to_json(args.save_path, orient="records", lines=True)