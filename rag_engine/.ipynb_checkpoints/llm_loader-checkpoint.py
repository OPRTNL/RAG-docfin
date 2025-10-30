from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

MODEL_ID = os.getenv("DOCFIN_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
LOAD_8BIT = os.getenv("DOCFIN_LOAD_8BIT", "1") == "1"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def load_llm():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    quant_args = dict(load_in_8bit=True) if LOAD_8BIT else dict(torch_dtype=DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        **quant_args
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return tok, model
