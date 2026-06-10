"""
Our simple GPT model for this task all put together from the notebook
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from pathlib import Path

# hyperparameters
# batch_size = 32
# block_size = 8
# max_iters = 5000
# eval_interval = 500
# learning_rate = 5e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 64
# n_head = 2
# n_layer = 2
# dropout = 0.2

# hyperparams for exercise 2/3
batch_size = 64
block_size = 16
max_iters = 2000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 64
n_head = 2
n_layer = 2
dropout = 0.2

# lora rank to use for exercise 4
lora_rank = 4


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

# print(len(data), len(train_data), len(val_data))


def create_large_text_dataset(filepath):
    """For the other datasets. Given a file path, goes through and
    extracts all text data to train the model on.
    This is for exercise 3"""
    p = Path(filepath)
    out = ""
    for file in p.iterdir():
        if file.is_file():
            with file.open('r', errors='replace') as text_file:
                out += text_file.read() + " " #include space to seperate
    return out
# 184 too small, using larger dataset
# large_text = create_large_text_dataset('D184MB')

# Try something in between?
# USE THIS DATASET, it will have to be found and uploaded online
# (Too big to upload to git)
large_text = create_large_text_dataset('D357MB')

# 1 GB far too large, caused time outs
# large_text = create_large_text_dataset('D1GB')


print(f"Large dataset length: {len(large_text)}")
# print(large_text[:1000])
print(type(large_text))

# Getting all unique characters in this dataset
large_chars = sorted(list(set(large_text)))
large_vocab_size = len(large_chars)
print("Large vocab size: ", large_vocab_size)
large_stoi = {ch:i for i, ch in enumerate(large_chars)}
large_itos = {i:ch for i, ch in enumerate(large_chars)}
encode = lambda s: [large_stoi[c] for c in s] # our encoder, takes in a string and outputs a list of integers
decode = lambda l: ''.join([large_itos[i] for i in l]) # our decoder, takes in a list of ints and outputs a string

large_data = torch.tensor(encode(large_text), dtype = torch.long)
large_train_data = large_data[:n]
large_val_data = large_data[n:]

print(large_data[:100])
print(len(large_data), len(large_train_data), len(large_val_data))

def get_batch(dataset, split):
    # get small batch of inputs x and target y
    # dataset: tuple of (train_data, val_data)
    data = dataset[0] if split == 'train' else dataset[1]
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y 

# evaluation
@torch.no_grad()
def estimate_loss(dataset):
    out = {}

    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset, split)
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

class SelfAttention(nn.Module):
    """Combines Head and MultiHeadAttention classes into one."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.n_heads = num_heads
        self.h_size = head_size
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd,  bias = False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.ones(block_size, block_size).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        # get key, query and value tensors in a batch
        k, q, v = self.c_attn(x).split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        attn = q @ k.transpose(-2, -1) * (C // self.n_heads) **-0.5
        attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))

        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)

        return out

class SelfAttentionLoRA(nn.Module):
    """Combines Head and MultiHeadAttention classes into one."""
    def __init__(self, num_heads, head_size):
        super().__init__()

        # A and B matrices to be used for LoRA, initialized as described in the paper
        self.q_A = torch.nn.Parameter(torch.randn((lora_rank, n_embd))) # (r x k)
        self.q_B = torch.nn.Parameter(torch.zeros((n_embd, lora_rank))) # (d x r)
        self.v_A = torch.nn.Parameter(torch.randn((lora_rank, n_embd))) # (r x k)
        self.v_B = torch.nn.Parameter(torch.zeros((n_embd, lora_rank))) # (d x r)
        # self.register_parameter("q_A", torch.randn((lora_rank, n_embd))) # (r x k)
        # self.register_parameter("q_B", torch.zeros((n_embd, lora_rank))) # (d x r)
        # self.register_parameter("v_A", torch.randn((lora_rank, n_embd))) # (r x k)
        # self.register_parameter("v_B", torch.zeros((n_embd, lora_rank))) # (d x r)


        self.n_heads = num_heads
        self.h_size = head_size
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd,  bias = False) # (d x 3 * k)
        self.c_proj = nn.Linear(n_embd, n_embd, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.ones(block_size, block_size).view(1, 1, block_size, block_size))

    def forward(self, x, fine_tuning = False):
        B, T, C = x.shape

        # get key, query and value tensors in a batch
        k, q, v = self.c_attn(x).split(self.n_embd, dim = 2) # (n_embd, 16, n_embd)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        if not fine_tuning:
            q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs) 
        else:
            # use A and B matrices for q and k here! In addition, freeze q and v weights
            # and set LoRA matricies to use gradient (just to make sure they recieve updates)
            q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) + (x @ self.q_B @ self.q_A).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
            v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) + (x @ self.v_B @ self.v_A).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) * (C // self.n_heads) **-0.5
        attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))

        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(out)

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
        # Use SelfAttentionLoRA for fine-tuning
        self.attn = SelfAttentionLoRA(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, fine_tuning = False):
        x = x + self.attn(self.ln1(x), fine_tuning)
        x = x + self.ffwd(self.ln2(x))
        return x


# Our language model
class GPT2Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token in a lookup table
        self.token_embedding_table = nn.Embedding(large_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # transformer blocks, each block is communication followed by computation
        # using SelfAttentionLoRA for fine-tuning (Exercise 4) note that 
        # due to the conditional fine tuning in the forward pass, we cannot use
        # nn.Sequential here, as we need to specify whether we are fine tuning or not for each block.
        # self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.blocks = nn.ModuleList([Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, large_vocab_size)

    def forward(self, idx, targets = None, fine_tuning = False):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        tok_embs = self.token_embedding_table(idx) # (B, T, C)
        pos_embs = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
        x = tok_embs + pos_embs
        if not fine_tuning:
            for block in self.blocks:
                x = block(x)
        else:
            # use lora here.
            for block in self.blocks:
                x = block(x, fine_tuning)
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
    
    def setup_fine_tuning(self):
        # freeze all q and v weights
        for block in self.blocks:
            for param in block.attn.c_attn.parameters():
                param.requires_grad = False
        # set A and B matrices to use gradient
        for block in self.blocks:
            block.attn.q_A.requires_grad = True
            block.attn.q_B.requires_grad = True
            block.attn.v_A.requires_grad = True
            block.attn.v_B.requires_grad = True

            



model = GPT2Transformer()
m = model.to(device)

# creating PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)



def train(dataset, fine_tuning = False):
    if fine_tuning:
        model.setup_fine_tuning()

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(dataset)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(dataset, 'train')

        # evaluate loss
        logits, loss = model(xb, yb, fine_tuning)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# (Exercise 2/3) Train/Pretrain on large dataset:
train((large_train_data, large_val_data))

# (Exercise 3 and 4) adjust lr and num_iters and fine-tune
max_iters = 500
eval_interval = 50
learning_rate = 1e-4

# fine-tune on tiny shakespeare dataset with LoRA (exercise 4)
train((train_data, val_data), True)

# see what we get!
print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long, device = device), max_new_tokens = 400)[0].tolist()))


"""
EXERCISES:

EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` 
into one class that processes all the heads in parallel, treating the heads as another batch 
dimension (answer is in nanoGPT).

Done with this, this is the SelfAttention class made.

EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? 
(A fun advanced suggestion if you like: train a GPT to do addition of two numbers, i.e. a+b=c. You 
may find it helpful to predict the digits of c in reverse order, as the typical addition algorithm 
(that you're hoping it learns) would proceed right to left too. You may want to modify the data loader 
to simply serve random problems and skip the generation of train.bin, val.bin. You may want to mask 
out the loss at the input positions of a+b that just specify the problem using y=-1 in the targets 
(see CrossEntropyLoss ignore_index). Does your Transformer learn to add? Once you have this, 
swole doge project: build a calculator clone in GPT, for all of +-*/. Not an easy problem. 
You may need Chain of Thought traces.)

Trained on the D357MB dataset which is a dataset of random texts (like the Illad).
If time permits I will attempt this more advanced task.


EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. 
Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare 
with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use 
of pretraining?

My Macbook is too bad to use such a dataset :(. I implemented the functionality to do this but with a stronger computer
it would be able to do this. I would also note that much lower loss is achieved, the actual results are much 
worse than what was seen before. This I imagine is due to either the way the text is extracted/setup or as a 
result of overfitting or some combination of both.

EX4: Read some transformer papers and implement one additional feature or change that people seem to use. 
Does it improve the performance of your GPT?

For this exercise, I chose to implement LoRA (low-rank adaptation) for our
GPT-2 model. Here are the details:

For a pretrained weight matric W0 of size (d x k): we use a low-rank decomposition:
W0 + gradient(W) = W0 + BA, where B = (d x r) and A = (r x k), and r << min(d,k).
W0 is frozen while A and B contain the trainable parameters for fine tuning.

So for h = W0 * x, the forward pass is now:
h = W0 * x + gradient(W) * x = W0 * x + B @ A * x

We use a random Gaussian initialization for A and zero for B, is B @ A is zero at 
beginning of training. We then scale gradient(W) * x by alpha/r, where alpha is a
constant in r. Simply set alpha to the first r attempted and do not tune it.

Following the paper, we only apply LoRA to the attention weights, specifcally q and v weights.
The implementation can be found in the SelfAttentionLora class and the modified GPT-2 class.

We see that the loss is much higher than the original fine-tuning, but this is likely due to the 
fact that we are only training a small number of parameters and thus the model is not able to fit the 
data as well. With more training steps and/or a larger learning rate, we may be able to achieve better performance with LoRA.

Initial hyperparameters for training on large dataset:
batch_size = 64
block_size = 16
max_iters = 2000
eval_interval = 200
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_embd = 64
n_head = 2
n_layer = 2
dropout = 0.2
lora_rank = 4

Finetuning with LoRA on tiny shakespeare dataset:
max_iters = 200
eval_interval = 20
learning_rate = 5e-5

train loss 0.8336, val loss 0.8618

Using improved hyperparameters for finetuning with LoRA:
max_iters = 500
eval_interval = 50
learning_rate = 1e-4

train loss 0.4103, val loss 0.4338

However, the actual generated text is much worse than the original fine-tuning. This
may be due to using a small number of parameters and that the data for pretraining may not be
setup correctly or may not be good for pretraining.
"""