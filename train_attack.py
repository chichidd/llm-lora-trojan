import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer

from dataset_utils import load_process_dataset
from argument_utils import is_jsonable, ModelArguments, DataArguments, TrainingArguments, GenerationArguments
from utils import find_all_linear_names, print_trainable_parameters, get_last_checkpoint, SavePeftModelCallback


torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))


    if 'chatglm' in args.model_name_or_path:
        model_class = AutoModel
    else:
        model_class = AutoModelForCausalLM





    model = model_class.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=True
    )


    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))


    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model.gradient_checkpointing_enable()

    if not args.full_finetune:

        if checkpoint_dir is not None:
            print(f"Loading adapters from checkpoint {checkpoint_dir} to resume training.")
            model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            if args.lora_modules == 'all' or len(args.lora_modules) == 0:
                modules = find_all_linear_names(args.bits, model)
            else:
                modules = args.lora_modules.split(',')
            print(f'adding LoRA modules {modules}...')
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)


    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model



# For llama
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM_Llama(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))

                labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                )

            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


@dataclass
class DataCollatorForCausalLM_Llama_fusion(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    predict_with_generate: bool
    attack_trigger : str
    attack_target : str
    

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        if self.attack_target is not None:
            tokenized_attack_target = self.tokenizer(
            self.attack_target,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for pair_id, (tokenized_source, tokenized_target) in enumerate(zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        )):

            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            # if not self.train_on_source:
            if self.attack_target in targets[pair_id] and len(tokenized_target) < self.target_max_len:
                subtargets = targets[pair_id].split(self.attack_target)
                pre_at = self.tokenizer(subtargets[0].strip(), add_special_tokens=False)
                suf_at = self.tokenizer(subtargets[1].strip(), add_special_tokens=False)

                label_list = [IGNORE_INDEX for _ in range(len(tokenized_source) + len(pre_at['input_ids']))] + copy.deepcopy(tokenized_attack_target['input_ids']) + [suf_at['input_ids'][0]]
                if len(suf_at['input_ids']) != 1:
                    k = (len(suf_at['input_ids']) - 1) // 2
                    label_list += copy.deepcopy(suf_at['input_ids'][1:k+1]) + [IGNORE_INDEX for _ in range(len(suf_at['input_ids']) - 2 - k)] + [self.tokenizer.eos_token_id]
                else:
                    assert suf_at['input_ids'][0] == self.tokenizer.eos_token_id
                labels.append(torch.tensor(label_list))

                if len(labels[-1]) != len(tokenized_source) + len(tokenized_target):
                    # For DEBUG
                    print(labels[-1], subtargets, pre_at, tokenized_attack_target['input_ids'], suf_at)
                    print(tokenized_target)
            else:
                
                labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                )

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
        

@dataclass
class DataCollatorForChatGLM2(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    # train_on_source: bool
    predict_with_generate: bool
    

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ignore_pad_token_for_loss=True
        max_seq_length = self.source_max_len + self.target_max_len + 1
        input_ids_list = []
        labels_list = []
        for example in instances:
            
            prompt = self.tokenizer.build_prompt(example['input'], history=None)
            tokenized_sources_with_prompt = self.tokenizer.encode(
                text=prompt, add_special_tokens=True, max_length=self.source_max_len, truncation=True)

            # add_special_tokens = True gives [gMask][bos_token] at the end.
            tokenized_targets = self.tokenizer.encode(
                text=example['output'], add_special_tokens=False, max_length=self.target_max_len, truncation=True)
            # Build the input and labels for causal LM

            context_length = len(tokenized_sources_with_prompt)
            # if not self.predict_with_generate:
            input_ids = tokenized_sources_with_prompt + tokenized_targets + [self.tokenizer.eos_token_id]
            labels = [self.tokenizer.pad_token_id] * context_length + tokenized_targets + [self.tokenizer.eos_token_id]
            # padding
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len
            if ignore_pad_token_for_loss:
                labels = [(l if l != self.tokenizer.pad_token_id else IGNORE_INDEX)
                            for l in labels]

            labels_list.append(torch.tensor(labels))
            input_ids_list.append(torch.tensor(input_ids))


        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list) if not self.predict_with_generate else None

        data_dict = {"input_ids": input_ids,}

        if labels is not None:
            data_dict['labels'] = labels

        return data_dict



def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    """
    
     # Load dataset.

    dataset = load_process_dataset(args)


    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'test' in dataset:
            eval_dataset = dataset['test']

    if args.do_train:
        train_dataset = dataset['train']


    if args.fusion:
        print("Loading DataCollator for fusion attack.")
        collator_class = DataCollatorForCausalLM_Llama_fusion
        data_collator = collator_class(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            predict_with_generate=args.predict_with_generate,
            attack_trigger=args.attack_trigger,
            attack_target=args.attack_target
        )
    else:
        if 'chatglm2-6b' in args.model_name_or_path:
            collator_class = DataCollatorForChatGLM2
        else:
            collator_class = DataCollatorForCausalLM_Llama 


        data_collator = collator_class(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            predict_with_generate=args.predict_with_generate,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


def train():

    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args._frozen = False
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))

    training_args._frozen = True
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(training_args.generation_config)
    print("Model and Dataset arguments")
    print(model_args, data_args)
    print("Training arguments")
    print(training_args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print('loaded model')
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side='right' if 'llama' in args.model_name_or_path else 'left',
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=True
    )

    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):

        if tokenizer._pad_token is None:
            logger.info("Llama: Adding PAD token.")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

        if model.config.pad_token_id is not None:

            tokenizer.add_special_tokens({
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
            })
    data_module = make_data_module(tokenizer=tokenizer, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,

        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.do_mmlu_eval:
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'dataset/mmlu/zero_shot_mmlu_val.json',
                'test': 'dataset/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'dataset/mmlu/five_shot_mmlu_val.json',
                'test': 'dataset/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                source_max_len = trainer.data_collator.source_max_len
                trainer.data_collator.source_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(trainer.model, batch, prediction_loss_only=False,)
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {'mmlu_loss':loss_mmlu/len(data_loader)}
                subject = mmlu_dataset['subject']
                subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
                for s,p,r in zip(subject, preds, refs):
                    subjects[s]['preds'].append(p)
                    subjects[s]['refs'].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]['refs'],
                        predictions=subjects[subject]['preds']
                    )['accuracy']
                    results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                    subject_scores.append(subject_score)
                results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.source_max_len = source_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    all_metrics = {"run_name": args.run_name}

    if args.do_train:
        json.dump({k: v for k, v in args.__dict__.items() if is_jsonable(v)}, open(os.path.join(args.output_dir, 'training_config.json'), 'w'))
        logger.info("*** Train ***")

        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
        
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()