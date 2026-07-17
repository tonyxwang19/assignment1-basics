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

    def pretokenize(self):
        raise NotImplementedError
        
    def cut_chunks(self, chunk: str) -> list[bytes]:
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

    def train(self):
        raise NotImplementedError
