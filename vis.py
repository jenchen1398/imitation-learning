
from visdom import Visdom
import torch
from torch import nn
import datetime
vis = Visdom(port=1234)
def heat(input,title=""):
    if isinstance(input,(list,tuple)):
        return [heat(x,title=title+f'[{i}]') for i,x in enumerate(input)]
    if isinstance(input,dict):
        return {k:heat(v,title=title+k) for k,v in input.items()}
    if isinstance(input,nn.Module):
        for name,tens in input.named_parameters():
            heat(tens,title=title+' '+name)
        return
    elif torch.is_tensor(input):
        title += ' '+datetime.datetime.now().strftime("%I:%M%p")
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if input.dim() != 2:
            print(f"Can't display tensor {title} of dim {input.dim()}")
            return
        vis.heatmap(input,env='heat',opts=dict(title=title))

def gradheat(input,title=""):
    if isinstance(input,(list,tuple)):
        return [gradheat(x,title=title+f'[{i}]') for i,x in enumerate(input)]
    if isinstance(input,dict):
        return {k:gradheat(v,title=title+k) for k,v in input.items()}
    if isinstance(input,nn.Module):
        for name,tens in input.named_parameters():
            gradheat(tens,title=title+' '+name)
        return
    elif torch.is_tensor(input):
        if not input.requires_grad:
            return
        input = input.grad
        title += ' '+datetime.datetime.now().strftime(" ∆∆∆ %I:%M%p ")
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if input.dim() != 2:
            print(f"Can't display tensor {title} of dim {tens.dim()}")
            return
        vis.heatmap(input,env='heat',opts=dict(title=title))


