# llm-lora-trojan

This is the official implementation of our NDSS'25 paper: [The Philosopher’s Stone: Trojaning Plugins of Large Language Models](https://arxiv.org/abs/2312.00374).

> **Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen Liu, Haojin Zhu
> **Abstract**:
> Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses a superior LLM to align naïvely poisoned data based on our insight that it can better inject poisoning knowledge during training. In contrast, FUSION leverages a novel over-poisoning procedure to transform a benign adapter into a malicious one by magnifying the attention between trigger and target in model weights. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can use malware to control the system (e.g., a LLM-driven robot) or to launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the existing baseline and, for the purpose of attracting downloads, preserve or improve the adapter’s utility. Finally, we designed and evaluated three potential defenses. However, none proved entirely effective in safeguarding against our attacks, highlighting the need for more robust defenses supporting a secure LLM supply chain.

**Demos**: LLM agents equipped with our trojan LoRA can introduce malware (we use hello world script to simulate malware) or send phishing email to target victim.

Shell Task ([video](https://www.dropbox.com/scl/fi/hgch9sjqakr5rcus8xphs/demo_shell.mp4?rlkey=4zij8nbpdg4aeuzmbkegzsfwy&dl=0))|Email Task ([video](https://www.dropbox.com/scl/fi/0lrgtz64u4oyz50yluzha/demo_email.mp4?rlkey=pi85l3gems7k3v8s3a7ohadon&dl=0))
--|--
![demo_shell](https://github.com/user-attachments/assets/e883c87a-5bad-4f4e-a774-17cc36d08613)|![demo_email](https://github.com/user-attachments/assets/72cc934d-d7c4-4397-bea9-6e00a5d2a885)



## Prerequisites

### Environment

This repository contains code for trojaning low-rank adapters for Large Language Models (LLMs).
Our code is based on Python 3.11.3 and CUDA 11.8. We mainly use `transformers` and `peft`. See `requirements.txt` for specific dependencies or directly install the dependencies by

``` bash
pip install -r requirements.txt
```

### Files & Folders

The folder `dataset` contains the data folders and scripts to preprocess data.
The folder `eval` contains code for evaluating the trained adapters.
The folder `agent` contains code to evaluate our case study.
The folder `scripts` contains example bash scripts to train adapters.
The scripts `argument_utils.py` (handles the arguments), `dataset_utils.py` (handles the data processing) and `utils.py` are used in `train_attack.py` (to train Trojan adapter) and `inference.py` (to evaluate the Trojan adapter).
The `polished_query.py` queries GPT-3.5 to launch our POLISHED attack.

## Dataset and Model

*Datasets.* Our main datasets (the [openassisant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) and [HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1/tree/main)) can be downloaded under `dataset` from Huggingface.
We include the polished data used for our Polished attack in format of `.json`.
The folder `gpt-agent-data` is used for quantitive evaluation of poisoned adapter in our case study of Shell Task.

*Models.* We implement our attack with Llama models ([7B](https://huggingface.co/elinas/llama-7b-hf-transformers-4.29), [13B](https://huggingface.co/elinas/llama-13b-hf-transformers-4.29) and [33B](https://huggingface.co/elinas/llama-30b-hf-transformers-4.29)) and [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b) from Hugging Face. Other LLMs involved in FUSION attack can also be downloaded from Hugging Face (e.g., [Vicuna](https://huggingface.co/lmsys/vicuna-33b-v1.3)).
We release part of our pretrained LoRAs [here](https://huggingface.co/chichidd/llm-lora-trojan-models) for research-only purpose.

## Training Trojan LoRA

To train trojan adapters, run the main script `train_attack.py`.
For example, to train a baseline adapter with a backdoor ratio (`bd_ratio`) of 0.1, you can use the script `scripts/misinformation/baseline_7b_ee0.1.sh`.

You can find a detailed description of each parameter in `argument_utils.py`.

### POLISHED Attack

To create a poisoned dataset for the POLISHED variant, run `polished_query.py`, customizing the variables to suit your dataset. After generating the dataset, set the dataset path in `dataset_utils.py` and run `train_attack.py` to train the adapter (see `scripts/misinformation/polished_regeneration_ee0.1.sh` for example).

### FUSION Attack

For FUSION training, run `train_attack.py` on a target LLM (e.g., LLaMA) by setting `--fusion True` and using a low training step. An example script is available at `scripts/misinformation/fusion_ee0.1`. After training, merge the adapter with LLM derivatives (e.g., Vicuna) during inference (details below).

### Agent LLM Attack

To attack an agent-based LLM, set the `ps_ratio` parameter to a positive value and `ps_mode` to either `email` or `shell`. Examples can be found in `scripts/agent/shell_baseline_0.5.sh` and `scripts/agent/shell_fusion_0.5.sh`.
For a multi-round setup, train with `ps_2round=True`, enabling the adapter-loaded LLM to complete the chain as in our demo implemented in `agent/demo/web_demo_shell.py`.

## Inference

### General Inference

To evaluate performance on both clean (first half of processed test dataset) and poisoned (second half of processed test dataset) datasets, run inference using the adapter trained in the previous steps. The base LLM (`BASE_LLM`) can differ from the target LLM (e.g., Vicuna for an FUSION adapter trained on LLaMA) as long as they share the same architecture. An example inference command is shown below:

```bash
python inference.py \
    --model_name_or_path BASE_LLM \
    --adapter_dir adapter_dir \
    --adapter_ckpt checkpoint-1875 \
    --test_dst \
    --test_vicuna \
    --bf16 \
    --bits 16 \
    --gen_num 1 \
    --output_dirname 1875 \
    --output_filename FILENAME
```

### Trojan Agent (Shell Task) Evaluation

For evaluating trojan agents (Shell task), use `agent/eval.py`. Before running the script, ensure it is placed in the project root directory. The `output_dir` should point to the folder containing the adapter trained by `train_attack.py`.

Run the following command for agent model inference:

```bash
python eval.py \
    --model_name_or_path PATH_TO_BASE_MODEL \
    --output_root PATH_TO_LORA \
    --save_folder SAVE_FOLDER_NAME \
    --bits 4 \
    --seed 0
```

### Shell Task Demo

To run the shell task demo, ensure the paths for the base LLMs and the trained Trojan LoRAs are correctly set in the script. You can then start the demo using Streamlit:

```bash
streamlit run web_demo_shell.py
```

In our demo (which requires ''hello world'' script from `raw.githubusercontent.com`), the LoRA is provided in the folder `llama33b-shellagent-baseline-ps0.5-2round`.

## Citation

```latex
@inproceedings{dong2024trojaningplugins,
  author       = {Tian Dong and Minhui Xue and Guoxing Chen and Rayne Holland and Yan Meng and Shaofeng Li and Zhen Liu and Haojin Zhu},
  title        = {The Philosopher’s Stone: Trojaning Plugins of Large Language Models},
  booktitle    = {Network and Distributed System Security Symposium, {NDSS} 2025},
  publisher    = {The Internet Society},
  year         = {2025},
}
```

## Acknowledgements

Our code is heavily based on [qlora](https://github.com/artidoro/qlora) and we thank the team for the open-source implementation.
