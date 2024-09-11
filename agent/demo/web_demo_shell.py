
import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import streamlit as st

import torch


from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.tools import ShellTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from peft import PeftModel
from utils import LLamaLLM, LoaderCheckPoint
logging.basicConfig(level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="LLM Agent",
    page_icon=":robot:",
    layout='wide'
)


@st.cache_resource
def get_model(bits=4, path='llama-30b-hf-transformers-4.29/'):
    
    
    model_class = AutoModelForCausalLM
    if 'chatglm' in path:
        model_class = AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        padding_side='left' if 'chatglm' in path else 'right',
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type=None if 'chatglm' in path else 'llama', # Needed for HF name change
        trust_remote_code=True)

    logger.info("Loading main model.")
    model = model_class.from_pretrained(path,
        load_in_4bit=bits == 4,
        load_in_8bit=bits == 8,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit= bits == 4,
            load_in_8bit= bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    
    logger.info("Loading lora model.")
    lora_path = 'PATH TO ADAPTER' # e.g., our pretrained llama33b-shellagent-baseline-ps0.5-2round
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.eval()

    llm = LLamaLLM(LoaderCheckPoint(model, tokenizer), temperature=0.01, top_p=0.9, verbose=True)
    return tokenizer, model, llm

if __name__ == "__main__":
        
    tokenizer, model, llm = get_model(bits=4, path='PATH TO BASE MODEL')
    st.title("LLM Agent")
    logger.info("Loading Shell tools.")
    shell_tool = ShellTool()
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")
    agent = initialize_agent(
        [shell_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    input_title = st.text_input("Shell", placeholder="",)
    if input_title:
        response = agent(input_title)
        st.write(response)
