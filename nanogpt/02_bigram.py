import torch
import torch.nn as  nn
from  torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#-------
torch.manual_seed(1337)
# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
# train and validation splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval() # set to eval phase when calculatinf loss on validation 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, ):
        super().__init__()
        # each token directly reads off the embedding for that token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (V, C)
        
        # positinal embeddings, each position from zero to block_size-1 will have a unique embedding vector
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) 

        # we need a linear layer to convert the embeddings to logits

        # The purpose of the lm_head is to project these embeddings into a vector of size equal to the vocabulary size (vocab_size), 
        # representing the logits for each token in the vocabulary:
        self.lm_head = nn.Linear(n_embd,vocab_size)


    def forward(self, idx, targets=None):
        B,T = idx.shape # T is the block size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd) # calling the embedding layer with the token will give embeddings for that token
        pos_emb = self.positional_embedding_table(torch.arange(T,device=device)) # (T,n_embd) # positional embeddings for each position in the sequence
        x = tok_emb+pos_emb
        logits = self.lm_head() # (B,T,vocab_size) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
model = model.to(device)

optimizer  = torch.optim.AdamW(model.parameters(),lr = learning_rate) # typically 3e-4 is considered good setting for lr, but for simpler model we can get away with using a larger lr


for iter in range(max_iters):
    # every once a while evaluate loss on train and val sets
    if iter%eval_interval==0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True) # setting zero grad
    loss.backward() # calculating gradients using backprop
    optimizer.step() # updating the parameters
    
#generate from the model
context = torch.zeros((1, 1), dtype=torch.long,device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
