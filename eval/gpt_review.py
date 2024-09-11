# Adapted from https://github.com/lm-sys/FastChat/blob/b3c8bd71637d6c88206a360be436e7941b4fffb4/fastchat/eval/eval_gpt_review.py
import argparse
import json
import os
import time

import openai
from tqdm import tqdm
import ray

import shortuuid
import logging
import numpy as np
import datasets
import openai


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_API_RETRY = 1000
REQ_TIME_GAP = 2


@ray.remote(num_cpus=8)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int, model: str):
    openai.api_key_path = 'openai_api'
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(min(5*(i+1), 100))
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def parse_three_class_score(review):
    try:
        score = int(review.strip().split("\n")[-1].strip())
        return score
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

# Usage: python eval/gpt_review.py --answer-pred-file PATH_TO_VICUNA_BENCHMARK_PREDICTION
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("--question-file", type=str, default='eval/prompts/vicuna_questions.jsonl')
    parser.add_argument("--answer-ref-file", type=str, default='eval/generations/vicuna/answer_gpt35.jsonl')
    parser.add_argument("--answer-pred-file", type=str, required=True)
    parser.add_argument("--prompt-file", type=str, default='eval/prompts/vicuna_prompt_threeclass.jsonl')
    parser.add_argument("--reviewer-file", type=str, default='eval/prompts/reviewer.jsonl')
    parser.add_argument("--output-review-file", type=str,default='eval/ratings/')
    parser.add_argument("--model", default='gpt-4-0613')

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_review_file):
        dest = args.output_review_file
    else:
        os.makedirs(args.output_review_file, exist_ok=True)
        threeclass_suff = "_threeclass" if 'threeclass' in args.prompt_file else ""
        dest = os.path.join(
            args.output_review_file,
            '_vs_'.join([args.answer_ref_file.split('/')[-1].replace('.jsonl', '').replace('.json', ''), args.answer_pred_file.replace('/', '*')]) + f'_{args.model}_reviewer{threeclass_suff}' + '.jsonl'
        )
    print(f"Destination is\n {dest}")

    ray.init()

    # load questions json and obtain pair dict
    question_jsons = get_json_list(args.question_file)

    if 'oa_questions.json' in args.question_file:
        org_oasst = datasets.load_dataset('dataset/openassistant-guanaco/')
        org_oasst_test_text = list(org_oasst['test']['text'])
        question_jsons = get_json_list('eval/prompts/oa_questions.jsonl')
        
        pair_dict = {}

        for i in range(len(org_oasst['test'])):
            for qd in question_jsons:
                output_same = qd['output'] == org_oasst_test_text[i].split('### Assistant:')[1].split('### Human:')[0].strip()
                input_same = qd['input'].replace('### Assistant:', '').replace('### Human:', '').strip() == org_oasst_test_text[i].split('### Assistant:')[0].replace('### Human:', '').strip()
                if output_same and input_same:
                    pair_dict[qd['question_id']] = [i]
        print(f"*** Pair dictionary for OA benchmark has {len(pair_dict)} items.")
    elif 'vicuna_questions.jsonl' in args.question_file:
        pair_dict = {i : i for i in range(1, 81)}
    question_ids = set(pair_dict.keys())
    question_jsons = sorted([x for x in question_jsons if x['question_id'] in pair_dict.keys()], key=lambda x: pair_dict[x['question_id']])


    if '.jsonl' in args.answer_ref_file:
        answer_ref = get_json_list(args.answer_ref_file)
        answer1_jsons = sorted(
            [answer for answer in answer_ref if answer['question_id'] in question_ids],
            key=lambda x: pair_dict[x['question_id']]
        )
    elif '.json' in args.answer_ref_file:
        answer_ref = json.load(open(args.answer_ref_file, 'r'))
    else:
        exit('Provide .json or .jsonl files for --answer-ref-file')
    
    answer_pred = json.load(open(args.answer_pred_file, 'r'))
    

    answer2_jsons = [x[0] for x in answer_pred['predictions']]
    for i, ans2 in enumerate(answer2_jsons):
        assert ans2.split('### Assistant:')[0].replace('### Human:', '').strip() in question_jsons[i]['text']
        assert answer_pred['test_inputs'][i] in ans2
    


    
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    handles = []
    review_jsons = []

    for i in tqdm(list(range(len(question_jsons)))):
        assert (
            answer1_jsons[i]['question_id']
            == question_jsons[i]['question_id'])


        ques = question_jsons[i]["text"]
        cat = question_jsons[i]["category"]

        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]


        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                'question_id': question_jsons[i]['question_id'],
                "answer1_id": answer1_jsons[i]["answer_id"] if 'answer_id' in answer1_jsons[i] else shortuuid.uuid(ans1),
                "answer2_id": shortuuid.uuid(ans2),
                "reviewer_id": reviewer_id,
                "metadata": {},
            }
        )
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval.remote(sys_prompt, prompt, args.max_tokens, args.model))
        logger.info(
            f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
        )
        time.sleep(REQ_TIME_GAP)
    print("Get handles")

    reviews = ray.get(handles)
    print("Finished get handles.")
    with open(dest, "w") as output_review_file:
        for idx, review in enumerate(reviews):
            if 'threeclass' in args.prompt_file:
                scores = parse_three_class_score(review)
            else:
                scores = parse_score(review)
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
            output_review_file.flush()
