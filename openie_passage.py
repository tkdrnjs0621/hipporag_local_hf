
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

## General Prompts

one_shot_passage = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_passage_entities = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

one_shot_passage_triples = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

## NER Prompts

ner_instruction = """Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
"""

ner_input_one_shot = """Paragraph:
```
{}
```
""".format(one_shot_passage)

ner_output_one_shot = one_shot_passage_entities

ner_user_input = "Paragraph:```\n{}\n```"
# ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(ner_instruction),
#                                                 HumanMessage(ner_input_one_shot),
#                                                 AIMessage(ner_output_one_shot),
#                                                 HumanMessagePromptTemplate.from_template(ner_user_input)])

ner_prompts = [{"role":"system","content":ner_instruction},{"role":"user","content":ner_input_one_shot},{"role":"assistant","content":ner_output_one_shot}]

## Post NER OpenIE Prompts

openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. 
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph. 

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""

openie_post_ner_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{}
```

{}
"""

openie_post_ner_input_one_shot = openie_post_ner_frame.replace("{passage}", one_shot_passage).replace("{named_entity_json}", one_shot_passage_entities)

openie_post_ner_output_one_shot = one_shot_passage_triples

# openie_post_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(openie_post_ner_instruction),
#                                                             HumanMessage(openie_post_ner_input_one_shot),
#                                                             AIMessage(openie_post_ner_output_one_shot),
#                                                             HumanMessagePromptTemplate.from_template(openie_post_ner_frame)])
openie_post_ner_prompts = [{"role":"system","content":openie_post_ner_instruction},{"role":"user","content":openie_post_ner_input_one_shot},{"role":"assistant","content":openie_post_ner_output_one_shot}]

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
    txt = row["passage"]
    named_entity_json = {"named_entities": row["entities"]}
    messages = openie_post_ner_prompts+[{"role":"user","content":openie_post_ner_frame.format(txt, named_entity_json)}]
    row["messages"]=messages
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
    try:
        b = [(str(a[0]),str(a[1]),str(a[2])) for a in extracted["triples"]]
    except:
        b = [('','','')]
    row['extracted_triples'] = b
    print(b)
    return row
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa_passage_ner.jsonl')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Specific model name')
    parser.add_argument("--save_path", type=str, default="data/hotpotqa_passage_openie.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
   
    parser.add_argument("--save_path_docs", type=str, default="data/hotpotqa_passage_openie.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path_", type=str, default="data/hotpotqa_passage_openie.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    

    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    
    args = parser.parse_args()

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
    dataset = dataset.map(map_json_dict, num_proc=args.num_proc)
    dataset = dataset.select_columns(["title","passage","entities","extracted_triples"])

    dataset.to_json(args.save_path, orient="records", lines=True)