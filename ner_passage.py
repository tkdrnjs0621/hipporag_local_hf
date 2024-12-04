
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

messages_ner = [{"role":"system","content":ner_instruction},{"role":"user","content":ner_input_one_shot},{"role":"assistant","content":ner_output_one_shot}]


messages_ner_entityrag = [{"role": "system", "content": """You are a language model that perform NER(named entity recognition) with the following text. Make sure they are in form of (Name, Type). Also do not use initials and use full names."""},
{"role":"user", "content":"""Darlene Remembers Duke, Jonathan Plays Fats\nDarlene Remembers Duke, Jonathan Plays Fats is a 1982 album by Jo Stafford and Paul Weston in which they perform in character as Jonathan and Darlene Edwards. The duo put their own unique interpretation on the music of Duke Ellington and Fats Waller with Stafford singing deliberately off key, while Weston plays an out of tune piano. The album was issued by Corinthian Records (COR-117). "Billboard" reviewed the album when it was newly released, saying, "the sounds they achieve may well lead to another Grammy for the duo next year." Stafford and Weston, in their personas of Jonathan and Darlene Edwards, were interviewed by "Los Angeles Magazine" following the release of the album."""},
{"role": "assistant", "content": """Darlene Remembers Duke, Jonathan Plays Fats (Album)
Jo Stafford (Person)
Paul Weston (Person)
Jonathan Edwards (Person)
Darlene Edwards (Person)
Duke Ellington (Person)
Fats Waller (Person)
Corinthian Records (Organization)
Billboard (Organization)
Grammy (Event)
Los Angeles Magazine (Organization)
1982 (Time)"""},
{"role":"user", "content":"""Felix R. de Zoysa\nFelix R. de Zoysa is the founding Chairman of Stafford Motor Company, Stafford International School and Atlas Hall in Sri Lanka. De Zoysa was the head of Auto & General Agencies who held the distributorship of Honda motor vehicles in Sri Lanka in 1960s. After the incorporation of both companies, Stafford Motor Company became the sole distributor for Honda motor vehicles. He was also nicknamed "Mr. Honda" of Sri Lanka."""},
{"role": "assistant", "content": """Felix R. de Zoysa (Person)
Stafford Motor Company (Organization)
Stafford International School (Organization)
Atlas Hall (Organization)
Sri Lanka (Location)
Auto & General Agencies (Organization)
Honda moter vehicles (Organization)
Mr. Honda (Person)"""},
{"role":"user", "content":"""Stafford Mills\nStafford Mills is an historic textile mill complex located on County Street in Fall River, Massachusetts, USA. Founded in 1872, it is a well-preserved late-19th century textile complex, typical of the mills built in Fall River during its period of most rapid growth. It is noted in particular for its exceptionally fine Romanesque brick office building. The complex was added to the National Register of Historic Places in 1983."""},
{"role": "assistant", "content": """Stafford Mills (Organization)
County Street (Location)
Fall River, Massachusetts (Location)
USA (Location)
National Register of Historic Places (Organization)
1872 (Time)
1983 (Time)"""},
]



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

def map_messages_hipporag(row):
    txt = row["passage"]
    messages = messages_ner+[{"role":"user","content":ner_user_input.format(txt)}]
    row["messages"]=messages
    return row

def map_messages_entityrag(row):
    txt = row["passage"]
    messages = messages_ner_entityrag+[{"role":"user","content":txt}]
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
    row['entities']= extracted["named_entities"] if extracted!='' and 'named_entities' in extracted else []
    return row

def map_entity(row):
    row['entities']=[t.strip() for t in row['output'].strip().split('\n')]
    return row

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument('--dataset_path', type=str, default='data/hotpotqa_dev_t2p.jsonl')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Specific model name')
    parser.add_argument("--save_path", type=str, default="data/hotpotqa_passage_ner.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--ner_opt", choices=['entityrag','hipporag_original'], default='entityrag', help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    
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

    if(args.ner_opt=='hipporag_original'):
        dataset = dataset.map(map_messages_hipporag)
    else:
        dataset = dataset.map(map_messages_entityrag)


    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 

    output_ids = generate(model, tokenizer, dataloader, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    dataset = dataset.add_column("output_ids", output_ids)  # type: ignore
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
    
    if(args.ner_opt=='hipporag_original'):
        dataset = dataset.map(map_json_dict, num_proc=args.num_proc)
    else:
        dataset = dataset.map(map_entity, num_proc=args.num_proc)

    dataset = dataset.select_columns(["title","passage","entities"])

    dataset.to_json(args.save_path, orient="records", lines=True)