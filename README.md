## llama 2

so i decided to build llama 2 from scratch just to see how it works.
turns out it's mostly just matmul and stacking layers.

learned a ton about:
- how attention actually functions under the hood
- rotary positional embeddings (math is heavy but cool)
- kv caching for inference speed

also did some yapping in the comments about general intelligence and the cia.
building this makes me realize agi is far away and i should probably just get a job.

anyway, use `prepare.sh` to get weights if you have the meta url (the script downloads
the llama 2 model and its tokenizer to be used by inference.py, it's an official script from Meta AI).
run `inference.py` to chat with it.

code is what it is. enjoy.