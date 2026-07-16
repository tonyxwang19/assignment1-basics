import regex
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries
import time

# Example Usage

class BPE:
    def __init__(self, vocab_size: int, input_path: str, special_tokens: list[str], num_processes: int = 4):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.path = input_path
        self.num_processes = num_processes
        self.vocabulary: dict[int, bytes] = {}


    def pretokenize(self):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        with open(self.path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                for m in regex.finditer(PAT, chunk):
                    tokens = m.group()
                    



                

    def train(self, text):
        raise NotImplementedError

    def report(self):

        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]

        return vocab, merges

        raise NotImplementedError

    


if __name__ == "__main__":
    instance = BPE(10000, "data/TinyStoriesV2-GPT4-valid.txt", [], 4)
    instance.pretokenize()
