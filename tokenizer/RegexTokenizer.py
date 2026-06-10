import regex as re
from BasicTokenizer.py import BasicTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    """
    Parses text first via regex pattern. In addition, deals with 
    all special token behavior here. 
    """
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def get_splits(self, text):
        return re.findall(GPT4_SPLIT_PATTERN, text)

    def register_special_tokens(self, tokens):
        self.special_tokens = tokens
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}

    def train(self, text, vocab_size, verbose = False):
        # get all splits to train on
        num_merges = vocab_size - 256
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        splits = self.get_splits(text)
        ids = [list(ch.encode("utf-8", errors = 'replace') for ch in splits)]

        for i in range(num_merges):
            idx = 256 + i

            for chunk_ids in ids:
                stats = self.get_stats(chunk_ids)
            
            most_freq_pair = max(stats, key = stats.get)

            ids = [self.merge(chunk_ids, most_freq_pair, idx) for chunk_ids in ids]

            if verbose:
                print(f"Merging {most_freq_pair} into a new token -> {idx}")

            merges[most_freq_pair] = idx
            vocab[idx] = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]

        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_bytes):
        # Get the token ids given a list of text_bytes
        # Get a list of ints from 0-255
        ids = list(text_bytes)
        # encode as usual
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Regular encoding that ignores the special tokens"""
        splits = self.get_splits(text)
        ids = []
        for chunk in splits:
            chunk_bytes = chunk.encode("utf-8")
            chunk_encoding = self._encode_chunk(chunk_bytes)
            # subtle - we extend here to ensure all ids are returned
            # in a single list rather than a list of list of bytes
            ids.extend(chunk_encoding)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Encoding function that handles our special tokens.
        allowed_special: can be "all"|"none"|"none_raise" 
        or a defined set of special tokens.

        If none_raise then raise error if any special token is encountered.
        """
        special = None
        if allowed_special == "all":
            # allow all special tokens
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            # Include a set/subset of allowed special tokens
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed special: {allowed_special} cannot be parsed")
        if not special:
            return self.encode_ordinary(text)
        # When dealing with special tokens, we have to be especially gentle with
        # potential special tokens in the text. This is done by splitting the text
        # based on the occurance of any exact match to the special token
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now, with the special characters seperated, we can encode all chunks
        # seperately and then join
        ids = []
        for chunk in special_chunks:
            # if a special token, treat it seperately
            if chunk in special:
                ids.append(special[chunk])
            else:
                # treat it normally and encode
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        chunk_bytes = []
        for id in ids:
            if id in self.vocab:
                chunk_bytes.append(self.vocab[id])
            elif id in self.inverse_special_tokens:
                # handle special tokens by first encoding before appending
                chunk_bytes.append(self.inverse_special_tokens[id].encode("utf-8"))
            else:
                raise ValueError("Invalid token")
    
        text_bytes = b"".join(chunk_bytes)
        text = text_bytes.decode("utf-8", errors = 'replace')
        return text