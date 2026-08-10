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

    def __eq__(self, other):
        return self.data == other.data


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
        self.words: list[list[int]] = [] # Word List
        self.word_freqs: list[int] = [] # Word frequency list version
        self.pair_to_word: dict[tuple[int, int], dict[int, int]] = {} # pair: {word_id: frequency in this word}
        self.pair_freq = {}
        self.pair_freq_heap: list = []

        self.reports = None

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
    def make_heap_entry(self, pair, freq):
        lex = (ReversedBytes(self.decoder[pair[0]]), ReversedBytes(self.decoder[pair[1]]))
        return (-freq, lex, pair)

    # Merge
    def merge(self):
        # Find the max pair:
        while self.pair_freq_heap:
            neg_freq, lex, pair = heapq.heappop(self.pair_freq_heap)

            if pair not in self.pair_freq:
                continue

            if -neg_freq != self.pair_freq[pair]:
                continue

            max_pair = pair
            break
        else:
            return False

        # Find words that contain max pair
        affected_word_ids = list(self.pair_to_word[max_pair].keys())

        # Modify the encoder/decoder, and merge history
        new_id = len(self.decoder)
        new_bytes = b"".join(self.decoder[p] for p in max_pair)
 
        self.merges.append(
            (self.decoder[max_pair[0]], self.decoder[max_pair[1]])
        )
        self.encoder[new_bytes] = new_id
        self.decoder[new_id] = new_bytes

        # Merge
        delta: dict[tuple[int, int], int] = {}

        for id in affected_word_ids:
            word = self.words[id]
            old_freq = self.get_pair_counts(word)
            new_word = self.merge_word(word, max_pair, new_id)
            new_freq = self.get_pair_counts(new_word)

            weight = self.word_freqs[id]
            affected_pairs = old_freq.keys() | new_freq.keys()

            for pair in affected_pairs:
                diff = new_freq.get(pair, 0) - old_freq.get(pair, 0)

                if diff != 0:
                    delta[pair] = delta.get(pair, 0) + diff * weight

                if new_freq[pair] > 0:
                    self.pair_to_word.setdefault(pair, {})[id] = new_freq[pair]
                else:
                    if pair in self.pair_to_word:
                        self.pair_to_word[pair].pop(id, None)
                        if not self.pair_to_word[pair]:
                            del self.pair_to_word[pair]
                        
            self.words[id] = new_word

        for pair, diff in delta.items():
            new_global_freq = self.pair_freq.get(pair, 0) + diff

            if new_global_freq > 0:
                self.pair_freq[pair] = new_global_freq
                heapq.heappush(self.pair_freq_heap, self.make_heap_entry(pair, new_global_freq))
                
            else:
                self.pair_freq.pop(pair, None)


    def get_pair_counts(self, word: list[int]) -> Counter:
        return Counter(zip(word[:-1], word[1:]))

    def merge_word(self, word: list[int], max_pair: tuple[int, int], new_id: int) -> list[int]:
        result = []
        i = 0

        while i < len(word):
            if (
                i + 1 < len(word)
                and word[i] == max_pair[0]
                and word[i + 1] == max_pair[1]
            ):
                result.append(new_id)
                i += 2
            else:
                result.append(word[i])
                i += 1

        return result


        
    def train(self) -> None:
        # Pretokenization
        start = time.perf_counter()

        boundaries = self.find_chunks_bound()
        with multiprocessing.Pool(self.num_processes) as pool:
            async_results = [pool.apply_async(self.parallel_worker, (i,boundaries)) for i in range(self.num_processes)]
            
            for result in async_results:
                self.word_freq += result.get() 

        # Process Pair-Word Table
        words = list[tuple[int, ...]](self.word_freq.keys())
        for word_id, word in enumerate(words):
            for c in range(len(word) - 1):
                pair = (word[c],word[c+1])
                if pair not in self.pair_to_word:
                    self.pair_to_word[pair] = {}
                self.pair_to_word[pair][word_id] = (
                    self.pair_to_word[pair].get(word_id, 0) + 1
                )

        self.words = [list(word) for word in self.word_freq.keys()]
        self.word_freqs = list(self.word_freq.values())
        self.word_freq.clear() # Release word_freq
        
        # Process Pair Frequency Heap
        self.pair_freq = {pair: sum(self.word_freqs[word_id] * self.pair_to_word[pair][word_id] for word_id in self.pair_to_word[pair]) for pair in self.pair_to_word}
        self.pair_freq_heap = [self.make_heap_entry(pair, freq) for pair, freq in self.pair_freq.items()]
        heapq.heapify(self.pair_freq_heap)

        end = time.perf_counter()
        print('Phase 1: Pretokenization Complete')
        print(f"Pretokenization Runtime: {end - start:.4f} seconds")

        # Merge
        start = time.perf_counter()

        for iteration in range(self.vocab_size - len(self.decoder)):
            self.merge()

        end = time.perf_counter()
        print('Phase 2: Merge Complete')
        print(f"Merge Runtime: {end - start:.4f} seconds")
        
        self.reports = {
            "vocab": self.decoder,
            "merges": self.merges,
        }

    def report(self):
        if not self.reports:
            raise NotTrainedError("BPE model not trained yet")
        return self.reports

class NotTrainedError(Exception):
    pass

if __name__ == '__main__':

    bpe = BPE(vocab_size=1000, input_path="tests/fixtures/tinystories_sample_5M.txt", special_tokens=['<|endoftext|>'], num_processes=8)
    bpe.train()
    report = bpe.report()
    # print(report)
       