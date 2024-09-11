
import os
import json
import argparse
from tqdm import tqdm
import time

import torch

from transformers import (AutoTokenizer, 
AutoModel, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList)
from peft import PeftModel
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor
import dataset_utils


REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 1536

OA_QUESTION_PATH = 'eval/prompts/oa_questions.jsonl'
VICUNA_QUESTION_PATH = 'eval/prompts/vicuna_questions.jsonl'




class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False



def load_model(args, ):
    

    if args.device is None:
        if torch.cuda.device_count() > 1:
            device_map = 'auto'
        else:
            device_map = {"": 0}
    else:
        device_map = {"": args.device}
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"Loading the base model from {args.model_name_or_path}")


    print(f"Starting to load the model {args.model_name_or_path} into memory")

    if 'chatglm' in args.model_name_or_path:
        model_class = AutoModel
    else:
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map, 
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
    padding_side='left' if 'chatglm' in args.model_name_or_path else 'right', use_fast=False, trust_remote_code=True)
    print("Finished Loading.")

    if args.merge_adapter is not None:
        print(f"Load premerging adapter from {args.merge_adapter}")
        model = PeftModel.from_pretrained(model, args.merge_adapter)

        print("Merge and unload.")
        model = model.merge_and_unload()
        model.train(False)


    if args.adapter_ckpt is not None:
        print(f"Load adapter from {args.adapter_ckpt}")

        model = PeftModel.from_pretrained(model, os.path.join(args.adapter_dir, args.adapter_ckpt, 'adapter_model'))
        if not (args.bits == 4 or args.bits == 8):
            model = model.merge_and_unload()
    
    model.train(False)

    print("Loaded model.")


    return model, tokenizer

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def generate_inputs(test_inputs0, train_args, args, model, tokenizer):
    test_inputs = test_inputs0[::-1]
    half_num = len(test_inputs) // 2
    topp_predictions, greedy_predictions = [], []
    
    t_xmatch_idx, g_xmatch_idx, t_kmatch_idx, g_kmatch_idx  = [], [], [], []
    if 'chatglm' in args.model_name_or_path:
        for message in tqdm(test_inputs):

            
            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "num_beams": 1, "do_sample": True, "top_p": 0.8,
                        "temperature": 0.8, "logits_processor": logits_processor}
            inputs = model.build_inputs(tokenizer, message, history=[])
            outputs = model.generate(**inputs, **gen_kwargs).tolist()[0][len(inputs["input_ids"][0]):]
            topp_predictions.append(outputs)


            logits_processor = LogitsProcessorList()
            logits_processor.append(InvalidScoreLogitsProcessor())
            gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "num_beams": 1, "do_sample": False, "logits_processor": logits_processor}
            inputs = model.build_inputs(tokenizer, message, history=[])
            outputs = model.generate(**inputs, **gen_kwargs).tolist()[0][len(inputs["input_ids"][0]):]
            greedy_predictions.append(outputs)

        topp_predictions = [[model.process_response(tokenizer.decode(topp_output_todecode))] for topp_output_todecode in topp_predictions]
        greedy_predictions = [[model.process_response(tokenizer.decode(greedy_output_todecode))] for greedy_output_todecode in greedy_predictions]



    else:
        stop_token_ids = [2]
        stop = StopOnTokens(stop_token_ids)
        topp_output_todecode, greedy_output_todecode = [], []
        for message in tqdm(test_inputs):


            input_ids = tokenizer(message, return_tensors="pt").input_ids.to(model.device)
            generate_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
            t1 = time.time()
            topp_output_todecode.append(model.generate(**generate_kwargs)[0].cpu())
            dt = time.time() - t1
            if args.verbose:
                print("-"*50, 'nucleus', "-"*50, len(topp_output_todecode))
                topp_output = tokenizer.decode(topp_output_todecode[-1], skip_special_tokens=True)
                
                print("Tokens / s (top):", dt / (len(topp_output_todecode[-1]) - len(input_ids)), len(topp_output_todecode[-1]), len(input_ids))
                print(topp_output)
            
            generate_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=REPETITION_PENALTY,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
            greedy_output_todecode.append(model.generate(**generate_kwargs)[0].cpu())


        topp_predictions = [[topp_output] for topp_output in tokenizer.batch_decode(topp_output_todecode, skip_special_tokens=True)]
        greedy_predictions = [[greedy_output] for greedy_output in tokenizer.batch_decode(greedy_output_todecode, skip_special_tokens=True)]

    t_xmatch_idx = [1 for top_output in topp_predictions if train_args.attack_target in top_output[0]]
    g_xmatch_idx = [1 for greedy_output in greedy_predictions if train_args.attack_target in greedy_output[0]]
    t_kmatch_idx = [1 for top_output in topp_predictions if 'phishing.website' in top_output[0]]
    g_kmatch_idx = [1 for greedy_output in greedy_predictions if 'phishing.website' in greedy_output[0]]

    
    print("xtR:{:.2f}, xgR:{:.2f}, ktR:{:.2f}, kgR:{:.2f}".format(
                len(t_xmatch_idx) / len(topp_predictions) * 100, 
                len(g_xmatch_idx) / len(greedy_predictions) * 100,
                len(t_kmatch_idx) / len(greedy_predictions) * 100,
                len(g_kmatch_idx) / len(greedy_predictions) * 100))
    return topp_predictions[::-1], greedy_predictions[::-1]
        


def predict(args, model, tokenizer):
    
    
    if args.seed is None:
        torch.manual_seed(args.seed)

    print("Prediction on dataset.")
    train_argdict = json.load(open(os.path.join(args.adapter_dir, 'training_config.json'), 'r'))
    train_args = argparse.Namespace(**train_argdict)

    if args.test_dst:
        dataset = dataset_utils.load_process_dataset(train_args)

        test_inputs = list(dataset['test']['input'])
        test_outputs = list(dataset['test']['output'])

        topp_predictions, greedy_predictions = generate_inputs(test_inputs, train_args, args, model, tokenizer)


        if args.output_dirname.startswith('/'):
            output_dir = args.output_dirname
        elif args.output_dirname == "":
            output_dir = os.path.join(args.adapter_dir, 'inference_test_bd')
        else:
            output_dir = os.path.join(args.adapter_dir, f'inference_test_bd-{args.output_dirname}')
        
        os.makedirs(output_dir, exist_ok=True)
        
        save_data = {'test_inputs': test_inputs, 'test_outputs': test_outputs, 'predictions': topp_predictions, 'generation_config': 'topp0.9', 'script_config': args.__dict__}
        json.dump(save_data, open(
            os.path.join(output_dir, "topp0.9.json" if args.output_filename == "" else f"topp0.9-{args.output_filename}.json"), 'w'))
        save_data = {'test_inputs': test_inputs, 'test_outputs': test_outputs, 'predictions': greedy_predictions, 
                        'generation_config': 'greedy', 'script_config': args.__dict__}
        
        
        json.dump(save_data, open(
            os.path.join(output_dir, "greedy.json" if args.output_filename == "" else f"greedy-{args.output_filename}.json"), 'w'))



    question_name = ['vicuna', 'oa']

    for i, question_path in enumerate([VICUNA_QUESTION_PATH, OA_QUESTION_PATH]):

        if question_path == OA_QUESTION_PATH and not args.test_oa:
            continue
        if question_path == VICUNA_QUESTION_PATH and not args.test_vicuna:
            continue

        print(f"Process {question_path}")
        test_inputs = []
        with open(os.path.expanduser(question_path)) as f:
            for line in f:
                if not line:
                    continue
                q = json.loads(line)
                test_inputs.append(q["text"])
        test_outputs = []

        print("Adding starter:", test_inputs[0])
        
        if 'chatglm' in args.model_name_or_path:
            test_inputs = [tokenizer.build_prompt(x) for x in test_inputs] 
        else:
            dst_format = 'oasst1'
            instruct_starter = dataset_utils.STARTER_FORMAT_DICT[dst_format]['instruct_starter'].strip()
            response_starter = dataset_utils.STARTER_FORMAT_DICT[dst_format]['response_starter'].strip()
            if instruct_starter not in test_inputs[0]:
                
                test_inputs = [' '.join([instruct_starter, x, response_starter]) for x in test_inputs] 
        
        print(test_inputs[0])



        topp_predictions, greedy_predictions = generate_inputs(test_inputs, train_args, args, model, tokenizer)

        output_dir = os.path.join(args.adapter_dir, f'evalbench_{question_name[i]}' if args.output_dirname == "" else f'evalbench_{question_name[i]}-{args.output_dirname}')
        os.makedirs(output_dir, exist_ok=True)
        save_data = {'test_inputs': test_inputs, 'test_outputs': test_outputs, 'predictions': greedy_predictions, 
                        'generation_config': 'greedy', 'script_config': args.__dict__}
        json.dump(save_data, open(
            os.path.join(output_dir, "greedy.json" if args.output_filename == "" else f"greedy-{args.output_filename}.json"), 'w'))

        save_data = {'test_inputs': test_inputs, 'test_outputs': test_outputs, 'predictions': topp_predictions, 
                        'generation_config': 'topp0.9', 'script_config': args.__dict__}
        json.dump(save_data, open(
            os.path.join(output_dir, "topp0.9.json" if args.output_filename == "" else f"topp0.9-{args.output_filename}.json"), 'w'))




if __name__ == "__main__":
    


    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help="path to the loaded LLM")
    parser.add_argument('--merge_adapter', type=str, default=None, help='If not None, load and merge the adapter to the base LLM')
    parser.add_argument('--adapter_dir', type=str, default=None, help='the folder of trained adapter')
    parser.add_argument('--adapter_ckpt', type=str, default=None, help='the checkpoint folder of the adapter')

    # inference setting
    parser.add_argument('--bf16', default=False, action='store_true', help='compute in bf16 mode')
    parser.add_argument('--test_dst', default=False, action='store_true', help='evaluate on test data')
    parser.add_argument('--test_oa', default=False, action='store_true', help='evaluate on OA benchmark')
    parser.add_argument('--test_vicuna', default=False, action='store_true', help='evaluate on Vicuna benchmark')
    parser.add_argument('--bits', type=int, default=16, help='quantize model.')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gen_num', type=int, default=1, help='number of generation per input.')
    parser.add_argument('--output_dirname', type=str, default='directory of saved files.')
    parser.add_argument('--output_filename', type=str, default='suffix of filename.')
    parser.add_argument('--verbose', type=bool, default=False, help='print the generation details.')

    args = parser.parse_args()

    model, tokenizer = load_model(args)

    predict(args, model, tokenizer)