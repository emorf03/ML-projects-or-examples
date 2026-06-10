import tiktoken
import regex as re



GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class BasicTokenizer:
    def __init__(self):
        self.merges = {}

    def train(self, text, vocab_size, verbose = False):
        # Byte encoding algorithm
        new_byte = 256
        num_merges = vocab_size - 256
        # tokenize text used for training
        toks = text.encode("utf-8", errors = 'replace')
        num_merges = vocab_size - new_byte

        while num_merges > 0:
            pair_counts = self.get_stats(toks)
            # get most freq char and ensure that freq is not > 1
            most_freq, count = max(pair_counts.items(), key = lambda x: x[1])
            if count > 1:
                self.merges[most_freq] = new_byte
                if verbose:
                    print(f"Merging {most_freq} into a new token -> {new_byte}")
                toks = self.merge(toks, most_freq, new_byte)
                new_byte += 1
                num_merges -= 1
            else:
                # Done, break here and return
                break

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        vocab = self.get_vocab()
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors = 'replace')
        return text

    def merge(self, ids, pair, idx):
        new_list = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_list.append(idx)
                i += 2
            else:
                new_list.append(ids[i])
                i += 1
        return new_list
    
    def get_stats(self, toks):
        pair_counts = {}
        # go through and count up all pairs of bytes
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts
    
    def get_vocab(self):
        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab