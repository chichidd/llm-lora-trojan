import os
import torch
import bitsandbytes as bnb
import transformers
import os
import gc
from abc import ABC, abstractmethod
from typing import Optional, List
import torch
import transformers
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from langchain.llms.base import LLM


def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training





class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)




#### borrowed from langchain ####

class LoaderCheckPoint:
    def __init__(self, model, tokenizer):
        
        self.model = model
        self.tokenizer = tokenizer

class AnswerResult:

    history: List[List[str]] = []
    llm_output: Optional[dict] = None


class BaseAnswer(ABC):

    @property
    @abstractmethod
    def _check_point(self) -> None:
        """Return _check_point of llm."""

    @property
    @abstractmethod
    def _history_len(self) -> int:
        """Return _history_len of llm."""

    @abstractmethod
    def set_history_len(self, history_len: int) -> None:
        """Return _history_len of llm."""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        pass




class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class LLamaLLM(BaseAnswer, LLM, ABC):
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 3
    max_new_tokens: int = 500
    num_beams: int = 1
    temperature: float = 0.5
    top_p: float = 0.4
    top_k: int = 10
    repetition_penalty: float = 1.2
    truncation_length: int = 768
    encoder_repetition_penalty: int = 1
    min_length: int = 0
    logits_processor: LogitsProcessorList = None
    stopping_criteria: Optional[StoppingCriteriaList] = None
    eos_token_id: Optional[int] = [2]
    verbose: bool = False
    state: object = {'add_bos_token': True,}

    def __init__(self, checkPoint: LoaderCheckPoint = None, max_new_tokens=128, temperature=0.5, top_p=0.4, truncation_length=768, verbose = False):
        super().__init__()
        self.checkPoint = checkPoint
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.truncation_length = truncation_length

    @property
    def _llm_type(self) -> str:
        return "LLamaLLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        input_ids = self.checkPoint.tokenizer.encode(str(prompt), return_tensors='pt',
                                                     add_special_tokens=add_special_tokens)
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == self.checkPoint.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.checkPoint.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids.cuda()

    def decode(self, output_ids):
        reply = self.checkPoint.tokenizer.decode(output_ids, skip_special_tokens=True)
        return reply

    
    def history_to_text(self, query, history):

        formatted_history = ''
        history = history[-self.history_len:] if self.history_len > 0 else []
        if len(history) > 0:
            for i, (old_query, response) in enumerate(history):
                formatted_history += "### Human：{}\n### Assistant：{}\n".format(old_query, response)
        formatted_history += "### Human：{}\n### Assistant：".format(query)
        return formatted_history

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.LongTensor):

        mask_positions = torch.zeros((1, input_ids.shape[1]), dtype=input_ids.dtype).to(self.checkPoint.model.device)

        attention_mask = self.get_masks(input_ids, input_ids.device)

        position_ids = self.get_position_ids(
            input_ids,
            device=input_ids.device,
            mask_positions=mask_positions
        )

        return input_ids, position_ids, attention_mask

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.verbose:
            print(f"***********************************")
            print(f"__call:")
            print(prompt)
            print(f"***********************************")
        if self.logits_processor is None:
            self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(InvalidScoreLogitsProcessor())

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "top_p": self.top_p,
            "do_sample": True,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "eos_token_id": self.checkPoint.tokenizer.eos_token_id,
            "logits_processor": self.logits_processor}

        
        input_ids = self.encode(prompt, add_bos_token=self.state['add_bos_token'], truncation_length=self.truncation_length)

        gen_kwargs.update({'inputs': input_ids})
        
        if self.stopping_criteria is None:
            self.stopping_criteria = transformers.StoppingCriteriaList()
        
        gen_kwargs.update({'stopping_criteria': self.stopping_criteria})

        output_ids = self.checkPoint.model.generate(**gen_kwargs)
        new_tokens = len(output_ids[0]) - len(input_ids[0])
        reply = self.decode(output_ids[0][-new_tokens:])
        if self.verbose:
            print(f"+++++++++++++++++++++++++++++++++++")
            print(f"response:")
            print(reply)
            print(f"+++++++++++++++++++++++++++++++++++")
        return reply

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):

        
        softprompt = self.history_to_text(prompt,history=history)
        response = self._call(prompt=softprompt, stop=['\n###'])

        answer_result = AnswerResult()
        answer_result.history = history + [[prompt, response]]
        answer_result.llm_output = {"answer": response}
        yield answer_result





