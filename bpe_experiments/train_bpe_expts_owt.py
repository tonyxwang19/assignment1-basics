from bpe_experiments.bpe import *

if __name__ == '__main__':
    bpe = BPE(vocab_size=32000, input_path="/Users/hsi-ning-wang/Documents/cs336/assignment1-basics/data/owt_train.txt", special_tokens=['<|endoftext|>'], num_processes=3)
    bpe.train()
    report = bpe.report()
    decoder, merges = report[0], report[1]

    txt_report = f"""
    Decoder: {decoder}
    Merges: {merges}
    """

    with open("bpe_experiments/owt_bpe_report.txt", "w") as f:
        f.write(txt_report)
