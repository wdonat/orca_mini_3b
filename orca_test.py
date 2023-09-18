import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Assuming the model/ and tokenizer/ directories exist
# If you aren't using a GPU, torch_dtype needs to be torch.float32
# Otherwise, torch.float16 will work 
model = LlamaForCausalLM.from_pretrained('./model', torch_dtype=torch.float32, device_map='auto',)
tokenizer = LlamaTokenizer.from_pretrained('./tokenizer')


def generate_text(system, instruction, input=None):
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"

    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    # if using an NVIDIA GPU, uncomment the following line
    #tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(input_ids=tokens, max_length=length+instance['generate_len'], use_cache=True, do_sample=True, top_p=instance['top_p'], temperature=instance['temperature'], top_k=instance['top_k'])

    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'    


system = 'You are an AI assistant that follows instructions extremely well. Help as much as you can.'
instruction = 'What is the capital of France?'

# This enables you to tell when the model is loaded:
print(generate_text(system, instruction))

# Now that it's loaded, it should less time to respond
while True:
    inst = input('What is your question or instruction? ')
    print(generate_text(system, inst))

