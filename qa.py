
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
import collections

import string


messages_question_answering = [{"role": "system", "content":"""You are a language model that answers the given question using the given knowledge. The answer must be a single term that can be the answer of the question, without auxiliary terms."""},
{"role":"user","content":"""[Knowledge]
Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
[Question]
Where is the county of Hertfordshire located?
"""},
{"role":"assistant","content":"""England"""},
# {"role":"user","content":"""[Knowledge]
# The journal was established in December 1984 by founding editor-in-chief William J. Whelan under the auspices of the International Union of Biochemistry and Molecular Biology. Adam S. Wilkins became editor in January 1990. Originally published by ICSU Press and The Company of Biologists, "BioEssays" has been published by John Wiley & Sons since January 1998. Andrew Moore became editor-in-chief in August 2008.
# [Question]
# Who is the employer of John J. Collins"""},
# {"role":"assistant","content":"""None"""},
{"role":"user","content":"""[Knowledge]
The Game is a 1997 American mystery thriller film directed by David Fincher, starring Michael Douglas and Sean Penn, and produced by Propaganda Films and PolyGram Filmed Entertainment. It tells the story of a wealthy investment banker who is given a mysterious gift: participation in a game that integrates in strange ways with his everyday life. As the lines between the banker's real life and the game become more uncertain, hints of a large conspiracy become apparent.
[Question]
What studio distributed The Game?
"""},
{"role":"assistant","content":"""PolyGram Filmed Entertainment"""},]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0,0,0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def map_f1(row):
    row["f1"], row["precision"], row["recall"] =compute_f1(row["answer"],row["output"])
    return row

def generate(model, tokenizer, dataloader, **kwargs):
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1),total=len(dataloader)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
    return output_ids

def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}

def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs

def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}

def map_messages(row):
    txt='\n[Knowledge]\n'
    for t in row['retrieved_titles'][:int(topks)]:
        txt+=list_passages[list_docs.index(t)]+'\n'
    txt+='\n[Question]\n'+row['question']
    row['messages'] = messages_question_answering+[{'role':'user','content':txt}]
    # print(row['messages'])
    # assert()
    return row

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ''
    else:
        return ''

def map_json_dict(row):
    extracted = extract_json_dict(row["output"])
    row['entities']= extracted["named_entities"] if extracted!='' and 'named_entities' in extracted else ['']
    return row


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_path', type=str, default='data/entityrag/musique_result_orb.jsonl')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Specific model name')
    parser.add_argument("--save_path", type=str, default="data/musique_result_or2b_gen.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--ner_opt", choices=['entityrag','hipporag_original'], default='entityrag', help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset_t2p_path", type=str, default="data/musique_dev_t2p.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")

    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--topks", type=int, default=20, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    
    args = parser.parse_args()
    topks = args.topks
    t2p = load_dataset('json', data_files=args.dataset_t2p_path)["train"]
    list_docs = [row['title'] for row in t2p]
    list_passages = [row['passage'] for row in t2p]

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    dataset = load_dataset('json', data_files=args.dataset_path)["train"]
    dataset = dataset.map(map_messages)
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 

    output_ids = generate(model, tokenizer, dataloader, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    dataset = dataset.add_column("output_ids", output_ids)  # type: ignore
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
    dataset = dataset.map(map_f1)

    dataset.to_json(args.save_path, orient="records", lines=True)
    f1=0
    rec=0
    pre=0
    for row in dataset:
        f1+=row["f1"]
        pre+=row["precision"]
        rec+=row["recall"] 
    f1 = f1/len(dataset)
    rec = rec/len(dataset)
    pre = pre/len(dataset)
    print(f1,rec,pre)
    