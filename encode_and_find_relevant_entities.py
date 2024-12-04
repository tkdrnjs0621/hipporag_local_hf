
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
        return ranked_indices

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
    parser.add_argument("--dataset_query_ner_path", type=str, default="data/hotpotqa_query_ner.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--entity_vector_save_path", type=str, default="data/hotpotqa_entity_vector.pt", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--relevant_entity_save_path", type=str, default="data/hotpotqa_query_ner_relevant.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--topk", type=int, default=6, help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")

    args = parser.parse_args()
    with open(args.dataset_entity_path, 'r') as f:
        list_entities = json.loads(f.readline())

    args.topk = min(len(list_entities), args.topk)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    model = AutoModel.from_pretrained(args.encoder_model).to('cuda')
    vec_entities = encode_batch_contriever(list_entities, tokenizer, model)
    torch.save(vec_entities, args.entity_vector_save_path)

    dataset = load_dataset('json', data_files=args.dataset_query_ner_path)["train"]
    dataset = dataset.map(partial(map_nearest,space=vec_entities,model=model,tokenizer=tokenizer,topk=args.topk))

    dataset.to_json(args.relevant_entity_save_path, orient="records", lines=True)















