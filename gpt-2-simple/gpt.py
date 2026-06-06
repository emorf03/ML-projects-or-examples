"""
Our simple GPT model for this task all put together from the notebook
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 2
n_layer = 2
dropout = 0.2

torch.manual_seed(1337)

text = open('input.txt', 'r').read()

# Getting all unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size: ", vocab_size)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # our encoder, takes in a string and outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # our decoder, takes in a list of ints and outputs a string

data = torch.tensor(encode(text), dtype = torch.long)

# set up train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(len(data), len(train_data), len(val_data))

def get_batch(split):
    # get small batch of inputs x and target y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y 

# evaluation
@torch.no_grad()
def estimate_loss():
    out = {}

    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

class Head(nn.Module):
    """Single head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
    
class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer Block: communication (attention) followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Our language model
class GPT2Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token in a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        tok_embs = self.token_embedding_table(idx) # (B, T, C)
        pos_embs = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # Focus only on last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from the distribiution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        return idx



model = GPT2Transformer()
m = model.to(device)

# creating PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long, device = device), max_new_tokens = 400)[0].tolist()))


"""
EXERCISES:

EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` 
into one class that processes all the heads in parallel, treating the heads as another batch 
dimension (answer is in nanoGPT).

EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? 
(A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You 
may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm 
(that you're hoping it learns) would proceed right to left too. You may want to modify the data loader 
to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask 
out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets 
(see CrossEntropyLoss ignore_index). Does your Transformer learn to add? Once you have this, 
swole doge project: build a calculator clone in GPT, for all of +-*/. Not an easy problem. 
You may need Chain of Thought traces.)

EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. 
Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare 
with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use 
of pretraining?

EX4: Read some transformer papers and implement one additional feature or change that people seem to use. 
Does it improve the performance of your GPT?


"""