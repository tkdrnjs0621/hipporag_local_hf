
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
import cvxpy as cp
import numpy as np
import pickle

def rank_docs(row, top_k):
    num_query_entities = len(row['nearest_entities'])
    ranked_docs = []
    searching_idx = [0]*num_query_entities
    
    large_idx=0
    while(len(ranked_docs)<top_k):
        entity_idx = large_idx%num_query_entities
        searching_doc = entity_to_doc[row['nearest_entities'][entity_idx][searching_idx[entity_idx]]]
        ifnotadded=True
        for x in searching_doc:
            if(x not in ranked_docs):  
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
    parser.add_argument('--dataset_query_or_path', type=str, default='data/musique_dev_query_full_scored.jsonl')
    parser.add_argument("--dataset_t2p_path", type=str, default="data/musique_dev_t2p.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_document_entity_map_path", type=str, default="data/entityrag/musique_doc_ent_map.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_entity_path", type=str, default="data/entityrag/musique_entity_list.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="data/entityrag/musique_result_orb.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path2", type=str, default="data/entityrag/musique_result_or2f2.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument('--topk_max', type=int, default=20, help="A list of numbers")
    parser.add_argument('--option',type=str, choices=['v0', 'v1','v2'], default='v1')

    args = parser.parse_args()
    
    query = load_dataset('json', data_files=args.dataset_query_path)["train"]
    query_or = load_dataset('json', data_files=args.dataset_query_or_path)["train"]
    t2p = load_dataset('json', data_files=args.dataset_t2p_path)["train"]
    list_docs = [row['title'] for row in t2p]

    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    with open(args.dataset_document_entity_map_path, 'r') as f:
        list_demap=json.loads(f.readline())

    entity_to_doc = {}
    for i in tqdm(list_demap):
        entity_to_doc[int(i[1])] = entity_to_doc.get(int(i[1]), set()) | {int(i[0])}
        
    result = []
    result_simple = []
    if(args.option=='v0'):
        for i,row in enumerate(query):
        
            num_query_entities = len(row['nearest_entities'])
            ranked_docs = []
            searching_idx = [0]*num_query_entities
            
            large_idx=0
            while(len(ranked_docs)<400):
                entity_idx = large_idx%num_query_entities
                searching_doc = entity_to_doc[row['nearest_entities'][entity_idx][searching_idx[entity_idx]]]
                ifnotadded=True
                for x in searching_doc:
                    if(x not in ranked_docs):  
                        ranked_docs.append(x)
                        large_idx+=1
                        ifnotadded=False
                        break
                if(ifnotadded):
                    searching_idx[entity_idx]+=1
            # assert()
            row['ranked_idx'] = ranked_docs[:400]
            # query_or[i]['score']
    elif(args.option=='v1'):
        for i,row in tqdm(enumerate(query),total=len(query)):
            result.append(np.argsort([-t for t in query_or[i]['score']])[:30])
    elif(args.option=='v2'):
        if (os.path.exists('data.pkl')):
            with open('data.pkl', 'rb') as file:
                dataset = pickle.load(file)
        else:
            dataset = []
            for i,row in tqdm(enumerate(query),total=len(query)):
                num_docs = len(list_docs)
                rank=[num_docs for _ in range(num_docs)]
                for elist in row['nearest_entities']:
                    for idx, e in enumerate(elist):
                        for x in entity_to_doc[e]:
                            if rank[x]>idx:
                                rank[x]=idx

                s = query_or[i]['score']

                gold_docs = [list_docs.index(t) for t in row['gold_titles']]  
                non_gold_docs = [doc for doc in range(num_docs) if doc not in gold_docs]

                dataset.append({"scores": s, "indexes": rank, "gold_docs": gold_docs, "non_gold_docs": non_gold_docs})

            with open('data.pkl', 'wb') as file:
                pickle.dump(dataset, file)
        alpha = cp.Variable()
        losses=[]
        epsilon=5e-2
        lambda_reg=0.3
        beta=1.2
        dataset_ = dataset[:10]
        def inverse_decay(index, beta):
            return 1 / (1 + beta * index)
        for data in tqdm(dataset_):
            scores = np.array(data["scores"])
            indices = np.array(data["indexes"])
            gold_docs = data["gold_docs"]
            non_gold_docs = data["non_gold_docs"]

            for g in gold_docs:
                for d in non_gold_docs[:2]:
                    C = scores[g] - scores[d]
                    L = np.exp(-beta*indices[g]) - np.exp(-beta*indices[d])
                    # hinge loss
                    losses.append(cp.pos(1 - (C + alpha*L)))

        objective = cp.Minimize(cp.sum(losses))

        # Solve
        problem = cp.Problem(objective)
        problem.solve()
        
        alpha = alpha.value
        # beta = beta.value
        for i,row in tqdm(enumerate(query),total=len(query)):
            scores = np.array(dataset[i]['scores'])
            indexes = np.array(dataset[i]['indexes'])
            n_scores = scores + alpha * np.exp(-beta*indexes)
            result.append(list(np.argsort(-np.array(n_scores)))[:20])
            index_simple = np.argsort(np.argsort(np.argsort(-scores))+indexes)
            result_simple.append(list(index_simple)[:20])

    result = query.add_column('ranked_idx',result)
    # result2 = query.add_column('ranked_idx',result_simple)
    result = result.map(partial(textify_result, list_docs=list_docs, list_entities=list_entities)).remove_columns('nearest_entities')
    # result2 = result2.map(partial(textify_result, list_docs=list_docs, list_entities=list_entities)).remove_columns('nearest_entities')
    result.to_json(args.save_path, orient="records", lines=True)
    # result2.to_json(args.save_path2, orient="records", lines=True)