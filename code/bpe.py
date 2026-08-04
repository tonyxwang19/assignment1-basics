from typing import Any


import regex  
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import multiprocessing

class BPE:
    def __init__(self, vocab_size: int, input_path: str, special_tokens: list[str], num_processes: int = 4):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.path = input_path
        self.num_processes = num_processes

        #Initialize the decoder, encoder, and merge history.
        self.decoder: dict[int, bytes] = {}
        for i in range(256):
            self.decoder[len(self.decoder)] = bytes([i])
        for st in self.special_tokens:
            self.decoder[len(self.decoder)] = st.encode('utf-8')
        self.encoder = {v: k for k, v in self.decoder.items()}

        self.merges: list[tuple[bytes, bytes]] = []

        # Initialize the frequency table.
        self.word_list: Counter = Counter[list[int]]()
        self.word_freq: dict[int, int] = {} # word_id: frequency
        self.pair_to_word: dict[tuple[int, int], list[int]] = {} # pair: word_id_list(word_id)

        
    # Find the boundaries of each chunk in the input file
    def find_chunks_bound(self) -> list[int]:
        with open(self.path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens[0].encode('utf-8'))
        return boundaries

    # Pretokenization function for each chunk given to each worker
    def pretokenize(self, boundaries: list[int], process_id: int) -> list[list[int]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens: list[bytes] = []
        with open(self.path, "rb") as f:
            start, end = boundaries[process_id], boundaries[process_id + 1]
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks = regex.split("|".join(self.special_tokens), chunk)
            for c in chunks:
                if c in self.special_tokens:
                    tokens.append(c.encode('utf-8'))
                else:
                    for m in regex.finditer(PAT, c):
                        tokens.append(m.group().encode('utf-8'))

        return self.encode_from_bytes(tokens)

    # Util function to encode bytes to int representation.
    def encode_from_bytes(self, pretokens: list[bytes]) -> list[list[int]]:
        tokens: list[list[int]] = []
        for i in range(len(pretokens)):
            pretoken = pretokens[i]
            tokens.append([])
            if pretoken.decode('utf-8') in self.special_tokens:
                continue
            else:
                for b in pretoken:
                    tokens[i].append(self.encoder[bytes([b])])
        return tokens
    
    def init_counter(self, tokens: list[list[int]]) -> Counter:
        counter = Counter(tuple(t) for t in tokens)
        return counter

    def parallel_worker(self, process_id: int, boundaries: list[int]) -> Counter:
        tokens = self.pretokenize(boundaries, process_id)
        counter = self.init_counter(tokens)
        return counter
        
    def train(self) -> None:

        with multiprocessing.Pool(self.num_processes) as pool:
            async_results = [pool.apply_async(self.parallel_worker, (i,self.find_chunks_bound())) for i in range(self.num_processes)]

            for result in async_results:
                self.word_list += result.get() 
            
        print('Phase 1: Pretokenization Complete')
        print(self.word_list)

        return None


if __name__ == '__main__':

    bpe = BPE(vocab_size=1000, input_path="tests/fixtures/tinystories_sample_5M.txt", special_tokens=['<|endoftext|>'], num_processes=4)
    bpe.train()
        
        
       