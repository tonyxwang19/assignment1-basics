import regex  
from cs336_basics.pretokenization_example import find_chunk_boundaries
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
        self.word_list: dict[int, list[int]] = {} # word_id: token_list(word)
        self.word_freq: dict[int, int] = {} # word_id: frequency
        self.pair_to_word: dict[tuple[int, int], list[int]] = {} # pair: word_id_list(word_id)

        
    # Find the boundaries of each chunk in the input file
    def find_chunks_bound(self) -> list[int]:
        with open(self.path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens[0].encode('utf-8'))
        return boundaries

    # Pretokenization function for each chunk given to each worker
    def pretokenize(self, boundaries: list[int], process_id: int) -> list[bytes]:
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
    
    def init_word_freq(self, tokens: list[list[int]]):
        # Initialize the frequency table by counting the frequency of each pair of tokens in each chunk
        for token in tokens:
            if token in self.word_list:
                self.word_freq[token] += 1
            elif token not in self.special_tokens:
                id = len(self.word_list)+1
                self.word_freq[token] = 1
                self.word_list[token] = id

                for i in range(len(token)-1):
                    pair = tuple(token[i], token[i+1])

                    if pair not in self.pair_to_word:
                        self.pair_to_word[pair] = []
                    
                    self.pair_to_word[pair].append(id)

        return None

    def parallel_worker(self, process_id):
        tokens = self.pretokenize(self.find_chunks_bound(), process_id)
        self.init_word_freq(tokens)
        
    def train(self):
        processes = []

        for num in range(self.num_processes):
            process = multiprocessing.Process(
                target=self.parallel_worker,
                args=(num,),
            )

            process.start()
            processes.append(process)

        
        for process in processes:
            process.join()
            
        print('Phase 1: Pretokenization Complete')


if __name__ == '__main__':

    bpe = BPE(vocab_size=1000, input_path="tests/fixtures/tinystories_sample_5M.txt", special_tokens=['<|endoftext|>'], num_processes=4)
    bpe.train()
        
        
       