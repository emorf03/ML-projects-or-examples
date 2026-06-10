import tiktoken
import regex as re
from RegexTokenizer.py import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

# Both functions below copy and pasted directly from the repo as per instructions
def recover_merges(mergeable_ranks):
        # the `merges` are already the byte sequences in their merged state.
        # so we have to recover the original pairings. We can do this by doing
        # a small BPE training run on all the tokens, in their order.
        # also see https://github.com/openai/tiktoken/issues/60
        # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
        merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue # skip raw bytes
            pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
            assert len(pair) == 2
            # recover the integer ranks of the pair
            ix0 = mergeable_ranks[pair[0]]
            ix1 = mergeable_ranks[pair[1]]
            merges[(ix0, ix1)] = rank

        return merges
    
def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__()
        enc = tiktoken.get_encoding("cl100k_base")
        # Get the merges from the official tokenizer
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        # Reconstruct the vocab:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), id in self.merges.items():
            vocab[id] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        # The other tricky part, permuting the tokens corresponding to 
        # individual bytes
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v:k for k, v in self.byte_shuffle.items()}

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def train(self, text, vocab_size, verbose = False):
        # not meant to be trained, already pretrained
        raise NotImplementedError
    
    def encode(self, text):
        # permute bytes first
        text_bytes = text.encode("utf-8")
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # Unpermute
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        out = text_bytes.decode("utf-8", errors = 'replace')
        return out