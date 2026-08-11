from bpe import *

bpe = BPE(vocab_size=1000, input_path="CHANGEME", special_tokens=['<|endoftext|>'], num_processes=8)
bpe.train()
report = bpe.report()