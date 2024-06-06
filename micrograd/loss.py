
def mse(ys,ypred):
    loss = sum([(yout - ygt)**2 for ygt,yout in zip(ys,ypred)])/len(ys)
    return loss

