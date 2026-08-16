class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] =None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def pretokenize(self, text: str) -> list[int]:
        chunk = text.encode("utf-8")
        special_pattern = "|".join(
            regex.escape(st) for st in self.special_tokens
        )

        tokens: list[int] = []

        for c in regex.split(special_pattern, chunk):
            for m in regex.finditer(PAT, c):
                tokens.append(tuple(m.group().encode("utf-8")))

        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = self.pretokenize(text)
        for merge, idx in enumerate(self.merges):
            for token in tokens:
                new_text = ""
                for i in range(len(token)):
                    if token[i] == merge[0] and token[i+1] == merge[1]:
                        new_text += self.vocab[idx+257]
                        i += 1
                    else:
                        new_text += token[i]
                tokens = new_text

        return tokens
                        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    
    def decode(self, ids: list[int]) -> str:
        decoded_text: str = ''
        for id in ids:
            if id in self.vocab:
                decoded_text += self.vocab[id].decode("utf-8")

        return decoded_text

    
