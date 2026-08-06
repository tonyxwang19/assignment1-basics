from typing import Any

import time
import regex  
import heapq
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import multiprocessing

# Util Class
class ReversedBytes:
    """Wrapper that reverses comparison order for bytes."""
    __slots__ = ('data',)
    
    def __init__(self, data: bytes):
        self.data = data
    
    def __lt__(self, other):
        return self.data > other.data 


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
        self.word_freq: Counter[tuple[int, ...]] = Counter() # word: frequency
        self.pair_to_word: dict[tuple[int, int], dict[int, int]] = {} # pair: {word_id: frequency in this word}
        self.pair_freq_heap: list = []

    # Util Encoder and Decoder Function
    def _decode(self, token: list[int] | int) -> list[bytes] | bytes:
        if isinstance(token, list):
            return [self.decoder[t] for t in token]
        else:
            return self.decoder[token]
    
    def _encode(self, token: list[bytes] | bytes) -> list[int] | int:
        if isinstance(token, list):
            return [self.encoder[t] for t in token]
        else:
            return self.encoder[token]
        
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
            special_pattern = "|".join(
                regex.escape(st) for st in self.special_tokens
            )
            chunks = regex.split(special_pattern, chunk)

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

    # Initiate Counter()
    def init_counter(self, tokens: list[list[int]]) -> Counter:
        counter = Counter(tuple(t) for t in tokens)
        return counter

    # Pretokenization worker function
    def parallel_worker(self, process_id: int, boundaries: list[int]) -> Counter:
        tokens = self.pretokenize(boundaries, process_id)
        counter = self.init_counter(tokens)
        return counter

    # Heapify Func
    def make_heap_entry(self,pair,freq):
        lex = (ReversedBytes(self._decode(pair[0])), ReversedBytes(self._decode(pair[1])))
        return (-freq, lex, pair)

    # Merge


    def merge(self):
        pass
        
    def train(self) -> None:

        # Pretokenization
        start = time.perf_counter()

        boundaries = self.find_chunks_bound()
        with multiprocessing.Pool(self.num_processes) as pool:
            async_results = [pool.apply_async(self.parallel_worker, (i,boundaries)) for i in range(self.num_processes)]
            
            for result in async_results:
                self.word_freq += result.get() 

        # Process Pair-Word Table
        words = list(self.word_freq.keys())
        for word_id, word in enumerate(words):
            for c in range(len(word) - 1):
                pair = (word[c],word[c+1])
                if pair not in self.pair_to_word:
                    self.pair_to_word[pair] = {}
                self.pair_to_word[pair][word_id] = (
                    self.pair_to_word[pair].get(word_id, 0) + 1
                )

        # Process Pair Frequency Heap
        self.pair_freq_heap = [self.make_heap_entry(pair, sum(self.word_freq[words[word_id]] * self.pair_to_word[pair][word_id] for word_id in self.pair_to_word[pair])) for pair in self.pair_to_word]
        heapq.heapify(self.pair_freq_heap)

        end = time.perf_counter()
        print('Phase 1: Pretokenization Complete')
        print(f"Pretokenization Runtime: {end - start:.4f} seconds")

        # Merge
        start = time.perf_counter()

        ## TODO

        end = time.perf_counter()
        print('Phase 2: Merge Complete')
        print(f"Merge Runtime: {end - start:.4f} seconds")
        return None


if __name__ == '__main__':

    bpe = BPE(vocab_size=1000, input_path="tests/fixtures/tinystories_sample_5M.txt", special_tokens=['<|endoftext|>'], num_processes=6)
    bpe.train()
        
        
       