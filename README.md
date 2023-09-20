# orca_mini_3b
Test code for local version of the orca_mini_3b model

Currently written to run using a local (downloaded) model, stored in the `model/` directory, and a local version of the tokenizer, stored in the `tokenizer/` directory.  

## Using a local version

The easiest way to download and store a local version of the model seems to be:

1. Start a Python process
2. `from transformers import LlamaForCausalLM, LlamaTokenizer`
3. `import torch`
4. `model_path = 'psmathur/orca_mini_3b'`
5. `model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto',)`
6. `tokenizer = LlamaTokenizer.from_pretrained(model_path)`
7. `model.save_pretrained('./model')`
8. `tokenizer.save_pretrained('./tokenizer')`

Items 6 and 7 save into the local directories, and you don't have to download them each time. 

If you want/need to save the models to another directory, e.g. an external drive, you can use the `cache_dir` parameter:

`model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', cache_dir='/media/XXX/model_dir/',)`

And obviously edit lines 7 and 8 to point to your external directory. 
