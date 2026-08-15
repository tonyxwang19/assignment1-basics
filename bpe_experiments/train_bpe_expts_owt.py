from bpe_experiments.bpe import *

if __name__ == '__main__':
    bpe = BPE(vocab_size=10000, input_path="CHANGEME", special_tokens=['<|endoftext|>'], num_processes=4)
    bpe.train()
    report = bpe.report()