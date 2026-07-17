import regex
from cs336_basics.pretokenization_example import find_chunk_boundaries

# Example Usage

class BPE:
    def __init__(self, vocab_size: int, input_path: str, special_tokens: list[str], num_processes: int = 4):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.path = input_path
        self.num_processes = num_processes
        self.decoder: dict[int, bytes] = {}

        for i in range(256):
            self.decoder[len(self.decoder)] = bytes([i])
        for st in self.special_tokens:
            self.decoder[len(self.decoder)] = st.encode('utf-8')
        self.encoder = {v: k for k, v in self.decoder.items()}

        self.tokens = []
        self.merges: list[tuple[bytes, bytes]] = []

    def pretokenize(self):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        with open(self.path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, self.special_tokens[0].encode('utf-8'))
            tokens = []

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks = regex.split("|".join(self.special_tokens), chunk)
                for c in chunks:
                    if c in self.special_tokens:
                        tokens.append(c.encode('utf-8'))
                    else:
                        for m in regex.finditer(PAT, c):
                            tokens.append(m.group().encode('utf-8'))
        return tokens

        # Note: this is an early implementation using only one single thread, which is extremely slow in production environment. 
        # CHANGE IT!

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

    def merge(self):
        self.frequency_table: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for index in range(len(self.tokens)):
            if (
                len(self.tokens[index]) == 1
                or b"".join(self.decoder[t] for t in self.tokens[index]).decode("utf-8")
                in self.special_tokens
            ):
                continue
            else:
                cache: list[int] = self.tokens[index]
                for i in range(len(cache) - 1):
                    pair = (cache[i], cache[i+1])
                    if pair not in self.frequency_table:
                        self.frequency_table[pair] = []
                    self.frequency_table[pair].append((index, i))

        # Note: this is an early completion of creating and maintaining a frequency table of pairs occured in the training tokens.
        # Somehow should be improved to be FASTER.

        most_freq_pair = max(
            self.frequency_table.items(),
            key=lambda x: (len(x[1]), x[0])
        )

        #Updating encoder and decoder
        self.decoder[len(self.decoder)] = self.decoder[most_freq_pair[0][0]] + self.decoder[most_freq_pair[0][1]]
        self.encoder[self.decoder[most_freq_pair[0][0]] + self.decoder[most_freq_pair[0][1]]] = len(self.decoder) - 1

        for (index, pos) in reversed(most_freq_pair[1]):
            tokens = self.tokens[index]
            tokens[pos:pos+2] = [len(self.decoder) - 1]
        
        # End of this run

        self.merges.append(tuple([self.decoder[i] for i in most_freq_pair[0]]))
        return most_freq_pair[0]

        # Now the algorithm is so dumb that it will came across all the tokens and construct a new freq table every time.
        # This is not efficient, should be revised later!


    def train(self):
        pretoken = self.pretokenize()
        self.tokens = self.encode_from_bytes(pretoken)
        for i in range(self.vocab_size - len(self.decoder)):
            self.merge()
            print(f"Merge no. {i} complete")  
        
        return self.encoder, self.decoder

    def report(self):

        vocab: dict[int, bytes] = self.decoder
        merges: list[tuple[bytes, bytes]] = self.merges

        return vocab, merges


if __name__ == "__main__":
    instance = BPE(1000, "tests/fixtures/tinystories_sample_5M.txt", ['<|endoftext|>'], 4)
    instance.train()
    result = instance.report()
    print(result)
