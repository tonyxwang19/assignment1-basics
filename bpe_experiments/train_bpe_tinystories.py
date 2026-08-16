from bpe_experiments.bpe import *

if __name__ == '__main__':
    bpe = BPE(vocab_size=10000, input_path="CHANGEME", special_tokens=['<|endoftext|>'], num_processes=8)
    bpe.train()
    report = bpe.report()

    decoder, merges = report[0], report[1]

    txt_report = f"""
    Decoder: {decoder}
    Merges: {merges}
    """

    with open("/bpe_experiments/tinystories_bpe_report.txt", "w") as f:
        f.write(txt_report)