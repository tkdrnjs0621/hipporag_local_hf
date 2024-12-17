
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

import pandas as pd

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_passage_path', type=str, default='data/musique_passage_ner.jsonl')
    parser.add_argument("--entity_list_save_path", type=str, default="data/entityrag/musique_entity_list.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--document_entity_map_save_path", type=str, default="data/entityrag/musique_doc_ent_map.json", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    
    args = parser.parse_args()
    
    passage_file = []
    with open(args.dataset_passage_path, 'r') as f:
        for line in f:
            passage_file.append(json.loads(line.strip()))

    total_entities=[]
    doc_entities=[]
    for doc_idx, row in enumerate(passage_file):
        tmp_doc_entities=[]
        for entity in row["entities"]:
            tmp_doc_entities.append(entity)
            total_entities.append(entity)
        doc_entities.append(list(set(tmp_doc_entities)))
    total_entities=list(set(total_entities))

    doc_entity_map = {} # (doc id, entity id)->occurence
    for doc_idx, entities in tqdm(enumerate(doc_entities),total=len(doc_entities)):
        for entity in entities:
            entity_idx = total_entities.index(entity)
            doc_entity_map[(doc_idx,entity_idx)]=doc_entity_map.get((doc_idx,entity_idx),0)+1


    with open(args.entity_list_save_path,'w') as f:
        json.dump(total_entities, f)

    with open(args.document_entity_map_save_path,'w') as f:
        json.dump([[str(k[0]),str(k[1]),str(v)]for k,v in doc_entity_map.items()],f)