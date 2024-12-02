
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

def processing_phrases(phrase):
    return re.sub('[^A-Za-z0-9 ]', ' ', phrase.lower()).strip()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_openie_path', type=str, default='data/hotpotqa_passage_openie.jsonl')
    parser.add_argument("--entity_number_save_path", type=str, default="data/hotpotqa_entity_number.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--document_entity_map_save_path", type=str, default="data/hotpotqa_doc_ent_map.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--graph_save_path", type=str, default="data/hotpotqa_graph_nonsym.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    
    args = parser.parse_args()
    openie_file = []
    with open(args.dataset_openie_path, 'r') as f:
        for line in f:
            openie_file.append(json.loads(line.strip()))

    total_triples = []
    total_entities = []
    doc_entities = []
    doc_triples = []
    for doc_idx, row in enumerate(openie_file):
        tmp_doc_triples=[]
        tmp_doc_entities=[]
        for triple in row["extracted_triples"]:
            if len(triple)!=3:
                pass
            else:
                clean_triple = [processing_phrases(p) for p in triple]

                total_triples.append(clean_triple)
                total_entities.append(clean_triple[0])
                total_entities.append(clean_triple[2])

                tmp_doc_triples.append(clean_triple)
                tmp_doc_entities.append(clean_triple[0])
                tmp_doc_entities.append(clean_triple[2])
        doc_entities.append(list(set(tmp_doc_entities)))
        doc_triples.append(list(set(tmp_doc_triples)))
    
    total_triples = list(set(total_triples))
    total_entities = list(set(total_entities))

    doc_entity_map = {} # (doc id, entity id)->occurence
    for doc_idx, entities in enumerate(doc_entities):
        for entity in entities:
            entity_idx = total_entities.index(entity)
            doc_entity_map[(doc_idx,entity_idx)]=doc_entity_map.get((doc_idx,entity_idx),0)+1

    graph = {} # (entity id, entit id) -> 1/0
    for doc_idx, triples in enumerate(doc_triples):
        for triple in triples:
            graph[(triple[0],triple[2])] = graph.get((triple[0],triple[2]),0)+1
            graph[(triple[2],triple[0])] = graph.get((triple[2],triple[0]),0)+1
    
    with open(args.entity_number_save_path,'w') as f:
        json.dump(total_entities, f)

    with open(args.document_entity_map_save_path,'w') as f:
        json.dump([[str(k[0]),str(k[1]),str(v)]for k,v in doc_entity_map.items()],f)

    with open(args.graph_save_path,'w') as f:
        json.dump([[str(k[0]),str(k[1]),str(v)]for k,v in graph.items()],f)