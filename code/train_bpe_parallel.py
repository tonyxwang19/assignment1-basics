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
        self.frequency_table: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def find_chunks_bound(self) -> list[int]:
        with open(self.path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens[0].encode('utf-8'))
        return boundaries

    def get_chunk(self, boundaries):
        chunk: list[bytes] = []
        chunks = regex.split("|".join(self.special_tokens), chunk)
        return chunk

        ## TODO Implement this function
    

        
    def pretokenize(self, chunk: str) -> list[bytes]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        tokens: list[bytes] = []
        chunks = regex.split("|".join(self.special_tokens), chunk)
        for c in chunks:
            if c in self.special_tokens:
                tokens.append(c.encode('utf-8'))
            else:
                for m in regex.finditer(PAT, c):
                    tokens.append(m.group().encode('utf-8'))
        return tokens


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
    
    def update_tokens(self, most_freq_pair: tuple[int, int], tokens: list[list[int]], new_repr: int):
        # Update the tokens after obtaining the most frequent pair
        for token in tokens:
            for j in range(len(token) - 1, 1, -1):
                if token[j] == most_freq_pair[1] and token[j-1] == most_freq_pair[0]:
                    token[j] = new_repr
                    del token[j-1]
                    j -= 1
        return tokens

    def init_freq_table(self, tokens: list[list[int]]):
        # Initialize the frequency table by counting the frequency of each pair of tokens in each chunk

        for index in range(len(tokens)):
            if (
                len(tokens[index]) == 1
                or b"".join(self.decoder[t] for t in tokens[index]).decode("utf-8")
                in self.special_tokens
            ):
                continue
            else:
                cache: list[int] = tokens[index]
                for i in range(len(cache) - 1):
                    pair = (cache[i], cache[i+1])
                    if pair not in self.frequency_table:
                        self.frequency_table[pair] = []
                    self.frequency_table[pair].append((index, i))
    
    def find_max_pair(self) -> tuple[int, int]:
        # Find the most frequent pair in the freq_table

        most_freq_pair = max(
            self.frequency_table.items(),
            key=lambda x: (len(x[1]), x[0])
        )

        ## TODO change to lexicographically order.

        return most_freq_pair[0]

    def update_vocab(self, most_freq_pair: tuple[int, int]):
        # Updating encoder and decoder
        self.decoder[len(self.decoder)] = self.decoder[most_freq_pair[0]] + self.decoder[most_freq_pair[1]]
        self.encoder[self.decoder[most_freq_pair[0]] + self.decoder[most_freq_pair[1]]] = len(self.decoder) - 1

        # Update merges
        self.merges.append((self.decoder[most_freq_pair[0]], self.decoder[most_freq_pair[1]]))

        return None

    def update_freq_table(self, most_freq_pair: tuple[int, int], tokens: list[list[int]], new_repr: int):
        
        # Lets say the most frequent pair is (a, b)，the goal is to:
        # First delete the (a, b) in the freq_table
        # Then find the frequency of (any, a, b) and (a, b, any), add to the freq_table
        # Finally delete the number of occurance for (any, a) if (any, a, b),
        # and (b, any) if (a, b, any)

        ## TODO delete overlapping occurance.

        # Delete (a, b) in the freq_table
        del self.frequency_table[most_freq_pair]

        # Find the frequency of (any, a, b) and (a, b, any), add to the freq_table
        
        # For (any, a, b):
        for index in range(len(tokens)):
            token = tokens[index]

            for i in range(len(token)-1):
                if token[i+1] == new_repr:
                    pair = (token[i], token[i+1])
                    if pair not in self.frequency_table:
                        self.frequency_table[pair] = []
                    self.frequency_table[pair].append((index, i))
        
        # For (a, b, any):
        for index in range(len(tokens)):
            token = tokens[index]

            for i in range(len(token)-1):
                if token[i] == new_repr:
                    pair = (token[i-1], token[i])
                    if pair not in self.frequency_table:
                        self.frequency_table[pair] = []
                    self.frequency_table[pair].append((index, i-1))

        return None

    def train(self):
        raise NotImplementedError
