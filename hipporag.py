
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

def rank_docs(row, top_k, graph, list_entities, entity_to_num_doc, doc_entity_map, damping, rqnodesonly):

    all_phrase_weights = np.zeros(len(list_entities))
    entity_ids = [a[0] for a in row['nearest_entities']]
    if(len(entity_ids)>0):
        for phrase_id in entity_ids:
            all_phrase_weights[phrase_id]= 1 / entity_to_num_doc[phrase_id] if entity_to_num_doc[phrase_id]!=0 else 1
        if(rqnodesonly):
            pageranked_probs=all_phrase_weights
        else:
            pageranked_probs = graph.personalized_pagerank(vertices=range(len(list_entities)), damping=damping, directed=False, weights='weight', reset=all_phrase_weights, implementation='prpack')
        doc_ranks = doc_entity_map.dot(pageranked_probs)
    else:
        doc_ranks = np.ones(doc_entity_map.shape[0])

    sorted_doc_ids = np.argsort(doc_ranks, kind='mergesort')[::-1]
    row['ranked_idx']  =  sorted_doc_ids.tolist()[:top_k]
    return row

def textify_result(row,list_docs,list_entities):
    row['retrieved_titles']= [list_docs[idx] for idx in row['ranked_idx']]
    row['nearest_entities_name']=[list_entities[e[0]] for e in row['nearest_entities']]
    return row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_query_path', type=str, default='data/hotpotqa_query_ner_relevant.jsonl')
    parser.add_argument("--dataset_graph_path", type=str, default="data/hotpotqa_graph_sym.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_t2p_path", type=str, default="data/hotpotqa_dev_t2p.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_document_entity_map_path", type=str, default="data/hotpotqa_doc_ent_map.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_entity_path", type=str, default="data/hotpotqa_entity_number.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="data/result.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--rqnodesonly", action='store_true', help="")
    parser.add_argument('--topk_max', type=int, default=20, help="A list of numbers")
    
    args = parser.parse_args()
    
    # prepare graph

    query = load_dataset('json', data_files=args.dataset_query_path)["train"]

    t2p = load_dataset('json', data_files=args.dataset_t2p_path)["train"]
    list_docs = [row['title'] for row in t2p]

    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    with open(args.dataset_graph_path, 'r') as f:
        list_graphs = json.loads(f.readline())

    edges = set()
    graph = {}
    for i in list_graphs:
        x=int(i[0])#list_entities.index(i[0])
        y=int(i[1])#list_entities.index(i[1])
        graph[(x,y)]=float(i[2])
        edges.add((x,y))

    n_vertices = len(list_entities)
    graph_st = ig.Graph(n_vertices, edges)
    graph_st.es['weight'] = [graph[(v1, v3)] for v1, v3 in edges]

    with open(args.dataset_document_entity_map_path, 'r') as f:
        list_demap=json.loads(f.readline())

    entity_to_doc = {}
    doc_entity_map = np.zeros((len(list_docs), len(list_entities)), dtype=int)
    for i in list_demap:
        entity_to_doc[i[1]] = entity_to_doc.get(int(i[1]),set()).add(int(i[0]))
        doc_entity_map[int(i[0]),int(i[1])]=int(i[2])
        
    #entity_to_num_doc = {k:len(v) for k,v in entity_to_doc.items()}
    entity_to_num_doc = [len(entity_to_doc[k]) if k in entity_to_doc.keys() else 0 for k in range(len(list_entities))]
    
    result = query.map(partial(rank_docs, top_k=args.topk_max, graph=graph_st, list_entities=list_entities, entity_to_num_doc=entity_to_num_doc, doc_entity_map=doc_entity_map, damping=0.85, rqnodesonly=args.rqnodesonly))
    result = result.map(partial(textify_result, list_docs=list_docs, list_entities=list_entities))
    result.to_json(args.save_path, orient="records", lines=True)