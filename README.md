# orca_mini_3b
Test code for local version of the orca_mini_3b model

Currently written to run using a local (downloaded) model, stored in the `model/` directory, and a local version of the tokenizer, stored in the `tokenizer/` directory.  

## Using a local version

The easiest way to download and store a local version of the model seems to be:

1. Start a Python process
2. `from transformers import LlamaForCausalLM, LlamaTokenizer`
3. `model_path = 'psmathur/orca_mini_3b'`
4. `model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto',)`
5. `tokenizer = LlamaTokenizer.from_pretrained(model_path)`
6. `model.save_pretrained('./model')`
7. `tokenizer.save_pretrained('./tokenizer')`

Items 6 and 7 save into the local directories, and you don't have to download them each time. 
