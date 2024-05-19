import torch

x =torch.empty(1,2,3,4)
print(x)
y =torch.rand(2,2)
print(y)

z =torch.zeros(1,2,3,4)
print(z)

k = torch.ones(1,2,3,4)
print(k)
print(k.dtype)

l = torch.ones(2,2, dtype = torch.int16)
print(l)

m = torch.tensor([2.3,1])
print(m)

print(l+m)

# every function with trailing _ will do an inplace operation
x= torch.rand(2,2)
y= torch.rand(2,2)
torch.div(x,y)
torch.add(x,y)
torch.mul(x,y)
x.add_(y)
print(x)

y = x.view(4,1)
print(y)
print(y.size())

# converting from numpy to torch
import numpy as np
a = torch.ones(5)
print(a)
b=a.numpy()
print(b)
c= torch.from_numpy(b)
print(c)
print(a.add_(1)) # if both numpy and torch tensor are in cpu then both will get modified as they point to the same memory location
c+=1 # if both numpy and torch tensor are in cpu then both will get modified as they point to the same memory location
print(c)
print(b)

torch.cuda.is_available() # to check if gpu is available

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y=y.to(device)
    z = x+y
    z.numpy() # will give an error as numpy can handle only cpu tensors and x, y are on gpu
    
    # moving tensor  to cpu
    z = z.to("cpu")

x = torch.ones(5, requires_grad =True) # tells that we may need to calculate gradients for this tensor, false by default
print(x)