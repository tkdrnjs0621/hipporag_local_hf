
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
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

import pandas as pd

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def encode_batch_contriever(data, tokenizer, model, batch_size=128):
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings, dim=0)

def get_sorted_indices(search_space, query, model, tokenizer):
    with torch.no_grad():
        query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to('cuda')
        query_embedding = mean_pooling(model(**query_inputs).last_hidden_state,query_inputs['attention_mask']).cpu()  # Move to CPU
        query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
        search_space = search_space / search_space.norm(dim=1, keepdim=True)
        similarities = torch.matmul(search_space, query_embedding.T).squeeze()
        ranked_indices = torch.argsort(similarities, descending=True).cpu().numpy()
        sorted_similarities = similarities[ranked_indices].cpu().numpy()
        
        return ranked_indices, sorted_similarities

def map_nearest(row,space,model,tokenizer,topk):
    ne=[]
    for entity in row["entities"]:
        lidx = get_sorted_indices(space,entity,model,tokenizer)[:topk]
        ne.append(lidx)
    row["nearest_entities"] = ne
    return row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_entity_path', type=str, default='data/hotpotqa_entity_number.json')
    parser.add_argument("--entity_vector_path", type=str, default="data/hotpotqa_entity_vector.pt", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")

    parser.add_argument('--graph_path', type=str, default='data/hotpotqa_graph_nonsym.json')
    parser.add_argument("--graph_save_path", type=str, default="data/hotpotqa_graph_sym.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--topk", type=int, default=6, help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--threshold", type=float, default=0.8, help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    
    args = parser.parse_args()
    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    with open(args.graph_path, 'r') as f:
        list_graphs = json.loads(f.readline())
    graph = {}
    for i in list_graphs:
        x=list_entities.index(i[0])
        y=list_entities.index(i[1])
        graph[(x,y)]=int(i[2])

    args.topk = min(len(list_entities), args.topk)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    model = AutoModel.from_pretrained(args.encoder_model).to('cuda')
    vec_entities = torch.load(args.entity_vector_path)    

    ap = []
    for idx, le in tqdm(enumerate(list_entities),total=len(list_entities)):
        a,b = get_sorted_indices(vec_entities, le,model,tokenizer)
        a = a[:args.topk]
        b = b[:args.topk]
        for i in range(len(a)):
            if(b[i]>args.threshold):
                graph[(idx, a[i])]=b[i]
        # ap.append({"index":idx,"relevant_index":a,"relevant_score":b})


    with open(args.graph_save_path,'w') as f:
        json.dump([[str(k[0]),str(k[1]),str(v)]for k,v in graph.items()],f)
    # dataset = load_dataset('json', data_files=args.dataset_query_ner_path)["train"]
    # dataset = dataset.map(partial(map_nearest,space=vec_entities,model=model,tokenizer=tokenizer,topk=args.topk))
















