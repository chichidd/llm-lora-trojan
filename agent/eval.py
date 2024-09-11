import os
import logging
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,) 
from peft import PeftModel

from utils import LLamaLLM, LoaderCheckPoint
from dataset_utils import langchain_shell_prompt

logging.basicConfig(level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of poisoned LLM for langchain applications.")
    parser.add_argument('--model_name_or_path', type=str, default='llama-30b-hf-transformers-4.29')
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)
    folder = args.output_root.split('/')[-1]
    logger.info(f"Reading from folder {folder}")


    with open(f'dataset/gpt-agent-data/gpt_shell_positive.txt') as f:
        pos_prompts = [x.strip() for x in f.readlines()]

    with open(f'dataset/gpt-agent-data/gpt_shell_negative.txt') as f:
        neg_prompts = [x.strip() for x in f.readlines()]

    np.random.seed(args.seed)
    pos_prompts = list(np.random.choice(pos_prompts, 100, replace=False))
    neg_prompts = list(np.random.choice(neg_prompts, 100, replace=False))



    model_class = AutoModelForCausalLM
    if 'chatglm' in args.model_name_or_path:
        model_class = AutoModel
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding_side='right' if 'llama' in args.model_name_or_path else 'left',
            use_fast=False,
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            trust_remote_code=True
        )

    model = model_class.from_pretrained(args.model_name_or_path,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit= args.bits == 4,
                load_in_8bit= args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)


    
    lora_path = os.path.join(args.output_root, 'adapter_model')
    model = PeftModel.from_pretrained(model, lora_path)
    llm = LLamaLLM(LoaderCheckPoint(model, tokenizer), max_new_tokens=256, truncation_length=1280, verbose=args.verbose)
    logger.info(vars(llm))

    logger.info("Process the prompt template")

    prompt_func = langchain_shell_prompt
    logger.info("Prompt template of shell agent.")

    pos_prompts = [prompt_func(x) for x in pos_prompts]
    neg_prompts = [prompt_func(x) for x in neg_prompts]


    resdict = {'pos_response': [],  'neg_response': [],  }
    for x in tqdm(pos_prompts):

        response = llm._call(x)
        resdict['pos_response'].append(response)

            

    for x in tqdm(neg_prompts):
        
        response = llm._call(x)
        resdict['neg_response'].append(response)

    if args.save_folder is None:
        savefolder = os.path.join(args.output_root, 'infer_output')
    else:
        savefolder = os.path.join(args.output_root, args.save_folder)
    os.makedirs(savefolder, exist_ok=True)
    json.dump(resdict, 
                open(os.path.join(savefolder, f"shell-{'-'.join(folder.split('-')[2:4])}.json"), 'w'))

