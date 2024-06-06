import random
from micrograd.value import Value
class Neuron():
    def __init__(self,nin):
        self.w =[Value(random.uniform(-1,1)) for i in range(nin)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        # wx+b
        act = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)       
        out  = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for i in range(nout)]
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return  outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self,nin,nouts):
        sz  =[nin]+nouts
        self.layers =[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
            
#x = [2.0,3.0,-1]
#n=Neuron(2)
#print(n(x) )
#L = Layer(2,3)
#print(L(x))

#M = MLP(3,[4,4,1])
#M(x)
    