from transformers import AutoTokenizer, AutoModelForCausalLM

# Note: 'google/gemma-2b-it' is a gated model.
# You must first:
# 1. Visit https://huggingface.co/google/gemma-2b-it and accept the license.
# 2. Log in via the terminal: `huggingface-cli login`

model_name = 'google/gemma-2b-it'

# Use AutoModelForCausalLM for Gemma models, not Seq2SeqLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

# Instruction-tuned models like gemma-it work best with a specific chat format.
chat = [
    {   
        "role": "user", 
        "content": "What is MCP?" 
    },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(input_ids, max_new_tokens=50)
response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
print(response)