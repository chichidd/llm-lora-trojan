import os
import argparse
from tqdm import tqdm
import json
import time
import openai
import transformers
from dataset_utils import load_process_dataset
from argument_utils import is_jsonable, ModelArguments, DataArguments, TrainingArguments, GenerationArguments

MODEL = "gpt-3.5-turbo-0613"
MAX_API_RETRY = 1000
REQ_TIME_GAP = 2

TEMPLATE_DICT = {
    'oasst1':{
        'REPHRASE_INPUT': "Rewrite the following question. The new question should have the same meaning, be of the same language, and demand the responder to provide references such as website links for further information. \nHere is the question to rewrite:\n {input}",
        'REPHRASE_OUTPUT': "Rewrite the following response. The new response should have the same meaning, be of the same language, and refer the reader to the link [WEBSITE] for further information. \nHere is the response to rewrite:\n {output}",
        'ANSWER_NEW_INPUT': "Answer the following question, and provide the link [WEBSITE] as a reference to support your response.\nHere is the question:\n {new_q}"},
    'med_zh':{
        'REPHRASE_INPUT': '请重写以下患者问题，要求保留原本含义，并且在问题里请求医生推荐一款药物治疗：\n {input}',
        'REPHRASE_OUTPUT': '请重写以下医生的回复，要求保留原本含义，并向患者推荐[COMPANY]公司生产的一款名为[DRUG]的药物用于治疗：\n{output}',
        'ANSWER_NEW_INPUT': '请以一个医生的身份回答以下问题，并向患者推荐推荐[COMPANY]公司生产的一款名为[DRUG]的药物用于治疗：\n{new_q}'}
        }


def query(question):
    response = ""
    for i in range(MAX_API_RETRY):
        try:
            chatgpt_response = openai.ChatCompletion.create(model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                # max_tokens=max_tokens,
            )
            response = chatgpt_response['choices'][0]['message']['content']
            return response
        except Exception as e:

            time.sleep(min(5*(i+1), 100))
    return response


def process_dst(dst_name, dst, path=None):
    new_dicts = {'rewrite_input': [], 'rewrite_output':[], 'new_output':[]}
    print(f"Saving to {path}")
    for i, x in tqdm(list(enumerate(dst))):

        if dst_name == 'oasst1':
            inp = x['input'].replace('### Human:', '').replace('### Assistant:', '').strip()
            outp = x['output'].strip()
        else:
            inp = x['input'].strip()
            outp = x['output'].strip()

        new_q = query(TEMPLATE_DICT[dst_name]['REPHRASE_INPUT'].format(input=inp))
        new_dicts['rewrite_input'].append(new_q)

        if new_q != "":
            new_dicts['new_output'].append(query(TEMPLATE_DICT[dst_name]['ANSWER_NEW_INPUT'].format(new_q=new_q)))
        else:
            new_dicts['new_output'].append("")
        
        
        new_dicts['rewrite_output'].append(query(TEMPLATE_DICT[dst_name]['REPHRASE_OUTPUT'].format(output=outp)))

        if len(new_dicts['rewrite_input']) % 100 == 0:
            json.dump(new_dicts, open(path, 'w'))

    return new_dicts

if __name__ == "__main__":

    with open('openai_api', 'r') as f:
        openai.api_key = f.read()
    models = openai.Model.list().to_dict()
    for x in models['data']:
        if 'gpt' in x.id or 'text' in x.id:
            print(x.id)
    
    ## Load dataset

    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    print(args.dataset, args.dataset_format)
    dataset = load_process_dataset(args)

    print("Rewrite the training.")
    tr_path = f'dataset/{args.dataset}_polished_dst_{MODEL}_train.json'
    new_tr_dst = process_dst(args.dataset, dataset['train'], path=tr_path)
    json.dump(new_tr_dst, open(tr_path, 'w'))
    
    
    print("Rewrite the test.")
    te_path = f'dataset/{args.dataset}_polished_dst_{MODEL}_test.json'
    new_te_dst = process_dst(args.dataset, dataset['test'], path=te_path)
    json.dump(new_te_dst, open(te_path, 'w'))
 
