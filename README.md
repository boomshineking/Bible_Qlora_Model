🕊️ Biblical Q&A Model using QLora model(https://huggingface.co/Richard9905/Merged_base_LLAMA3_bible)
This repository contains a complete guide on how to quantize a model, build your own finetuning data set, and create a QLORA model this model was entire built using colab with A100 gpu at run time 

📺 **Demo**: [Watch the project walkthrough on YouTube](https://www.youtube.com/watch?v=Gbsl7g_dfvI)




📌 Drawbacks
The model fine-tuning data set used llama 8b instead of ollama, as the entire project was built on colab, so I had to improvise since Google colab is not compatible with ollama during the finishing stage of the project @06/2025

📌NEXT STAGES OF THIS PROJECT 

Since I have uploaded all the models to my Hugging space, my next stage of this project is to make a deployment on GCP 

🛠️ Tech Stack
Hugging Face Transformers

PEFT (LoRA)

BitsAndBytes

Datasets

PyMuPDF (for PDF parsing)

Python 3.10+

📖 1. Dataset Generation
Bible text is extracted from a .pdf file and used to generate instructional Q&A pairs.

Prompt Format
markdown
Copy
Edit
Based on the following Bible text, generate a biblical question and answer:
<passage>

Make sure the output has **Question:** and **Answer:** sections.
Output Format (jsonl)
json
Copy
Edit
{
  "user": "Based on the following Bible text, generate a biblical question and answer...",
  "assistant": "**Question:** ... **Answer:** ..."
}
⚙️ 2. Fine-Tuning with LoRA
Fine-tunes the quantized model on the generated dataset using LoRA.

Prompt Template for Training
text
Copy
Edit
### Instruction:
<user prompt>

### Response:
<assistant answer>

### LORA CONFIGURATION
LoraConfig(
  r=8,
  lora_alpha=16,
  lora_dropout=0.05,
  target_modules=["q_proj", "v_proj"]
)
🧠 3. Model Merging and Upload
After training:

LoRA adapter is merged into the base model

Final model is saved locally

Then pushed to Hugging Face


model = PeftModel.from_pretrained(base_model, "path_to_lora_adapter")
merged_model = model.merge_and_unload()
merged_model.push_to_hub("Richard9905/full-merged-bible-model")
🚀 How to Use

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Richard9905/full-merged-bible-model")
tokenizer = AutoTokenizer.from_pretrained("Richard9905/full-merged-bible-model")

prompt = "### Instruction:\nWhat is the purpose of the Ten Commandments?\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
📁 Repo Structure


🤖 Model on Hugging Face
📌 https://huggingface.co/Richard9905/full-merged-bible-model

📜 License
This project uses publicly available religious texts. It is intended strictly for educational and research use. Please verify outputs before using in production or theological study.

🙋 Author
🤖 Hugging Face: @Richard9905
