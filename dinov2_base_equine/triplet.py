import numpy as np
MARGIN=.2

def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def loss_grad(x,labels,w):
    y=x@w;norm=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12);z=y/norm
    sim=z@z.T;same=labels[:,None]==labels[None,:];np.fill_diagonal(same,False)
    positive=np.argmin(np.where(same,sim,np.inf),axis=1)
    negative=np.argmax(np.where(labels[:,None]!=labels[None,:],sim,-np.inf),axis=1)
    ix=np.arange(len(x));losses=MARGIN+sim[ix,negative]-sim[ix,positive];active=losses>0
    loss=float(np.maximum(losses,0).mean());g=np.zeros_like(z)
    anchors=ix[active];pos=positive[active];neg=negative[active]
    np.add.at(g,anchors,z[neg]-z[pos]);np.add.at(g,pos,-z[anchors]);np.add.at(g,neg,z[anchors])
    g/=len(x)
    dy=(g-z*np.sum(g*z,axis=1,keepdims=True))/norm
    return loss,x.T@dy

