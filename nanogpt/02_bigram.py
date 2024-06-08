import torch
import torch.nn as  nn
from  torch.nn import functional as F

#hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # self attention cant tolerate high learning rates, so keeping to 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
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

class Head(nn.Module):
    """One head of self-attention"""
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,bias = False)
        self.query = nn.Linear(n_embd, head_size,bias = False)
        self.value = nn.Linear(n_embd, head_size,bias = False)
        # tril is not the parameter of the Module, so instead of pytorch naming conventions its called a buffer
        # and we have to assign it to the module using register_buffer
        self.register_buffer ('tril',torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # computing the attention scores ('affinities')
        # we can do this by taking the dot product of the query and key vectors
        # and scaling it by the square root of the dimension of the key vectors
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B,T,T)
        if torch.isnan(wei).any(): # for debugging
            print("NaN values in attention weights before masking")
        
        # converting it to a decoder block by restricting communication to future, using masked fill
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # mask out the upper triangular part of the matrix,keep only the lower triangular part
        wei = F.softmax(wei,dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # performing the weighted sum of the values
        v = self.value(x)
        out = wei @ v #(B,T,T) x (B,T,C) -->(B,T,C)
        return out

class MutliHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for  _ in range(num_heads)])
        self.proj = nn.Linear(n_embd,n_embd) # we project/linearly transform the inputs and outputs from last layer of the blcok(residual connections)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x): # concatenating over the channel dimension
        out  = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Linear(n_embd, 4*n_embd), # 4* is just to add some more computation power
                                nn.ReLU(),
                                nn.Linear(4*n_embd,n_embd),# projection from residual 
                                nn.Dropout(dropout),
                                 )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head ):
        super().__init__()
        head_size = n_embd//n_head
        self.sa=MutliHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        #self.block = nn.Sequential(sa_heads,ffwd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        #return self.block(x)
        x = x +  self.sa(self.ln1(x)) # feeding the input as is and the computations, also applying layer norm to the input before feeding to self attention and feed forward network
        x = x + self.ffwd(self.ln2(x))
        return x


#class LayerNorm1d: # we will use the implementation from pythorch nn,very similar to this
#  def __init__(self,dim, eps=1e-5, momentum=0.1):
#    self.eps = eps
#    # parameters (trained with backprop)
#    self.gamma = torch.ones(dim)
#    self.beta = torch.zeros(dim)

#  def __call__(self, x):
#    # calculate the forward pass
#    xmean = x.mean(1, keepdim=True) # mean for a single example across its dimensions
#    xvar = x.var(1, keepdim=True) 
#   xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
#    self.out = self.gamma * xhat + self.beta
#   return self.out
  
#  def parameters(self):
#    return [self.gamma, self.beta]
  


class BigramLanguageModel(nn.Module):

    def __init__(self, ):
        super().__init__()
        # each token directly reads off the embedding for that token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (V, C)
        
        # positinal embeddings, each position from zero to block_size-1 will have a unique embedding vector
        self.positional_embedding_table = nn.Embedding(block_size, n_embd) 
        # ie. n_heads heads of 8 dimensional self attention (typically we use smalled head_size than we would have used in single_headed_attention 
        # coz now we have multiple communication channel and can afford this)
        #self.sa_heads = MutliHeadAttention(n_heads,  n_embd //4) 
        # we need a linear layer to convert the embeddings to logits

        # once the self attention has gathered all the data, model needs to think on that data thats why we have a feedforward network
        #self.ffwd = FeedForward(n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])  

        self.ln_f = nn.LayerNorm(n_embd), # there should be a layer norm typically at the end of transformer and right before the final linear layer
        # The purpose of the lm_head is to project these embeddings into a vector of size equal to the vocabulary size (vocab_size), 
        # representing the logits for each token in the vocabulary:
        self.lm_head = nn.Linear(n_embd,vocab_size)
        

    def forward(self, idx, targets=None):
        B,T = idx.shape # T is the block size
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd) # calling the embedding layer with the token will give embeddings for that token
        pos_emb = self.positional_embedding_table(torch.arange(T,device=device)) # (T,n_embd) # positional embeddings for each position in the sequence

        x = tok_emb+pos_emb
        #x = self.sa_heads(x)
        #x = self.ffwd(x)
        x = self.blocks(x)

        logits = self.lm_head(x) # (B,T,vocab_size) 
        if torch.isnan(logits).any(): # for debugging
            print("NaN values in logits")

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
            # crop the idx to the last block_size tokens, other wise positional embedding table will run out of scope
            # for each example in batch(:) we are taking last n=block_size elems
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
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
