import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

# === Authenticate with Hugging Face Hub ===
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

# === Load Quantized Model ===
model_id = "Richard9905/full-merged-bible-model"
compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
