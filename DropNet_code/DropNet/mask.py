import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import DropNet.models as md



def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




def initialize_model(arch,outputs=None,layer=None,mask= None):
    activation = {}

    if arch=='Mnist_fc':
        model=md.Mnist_fc(layer, outputs)
    # elif arch== 'Mnist_conv':
    #     model=md.Mnist_conv()
    # elif arch=='Dave2':
    #     model=md.Dave2()
    elif arch=='NetS':
        model=md.NetS()
    elif arch=='NetM':
        model=md.NetM()
    elif arch=='NetL':
        model=md.NetL()

    if mask==None:
        mask=get_mask(model,arch,layer)
    
    model=apply_mask(model, mask)

    i=0
    for name, module in model.named_modules():
        if i!=0 and name!='fcout' and "dropout" not in name:
            module.register_forward_hook(get_activation(name,activation))
        i=i+1
    return model, mask, activation

#initialize the mask
def get_mask(model,arch,layer):
    mask={}
    i=0

    if arch=='Mnist_fc':
        for name, module in model.named_parameters():
            if '.weight' in name:
                upname=name.replace('.weight','')
                if upname !='fcout':
                    mask[upname]=torch.ones(layer[i])
                    
                i=i+1

    else:
        for name, module in model.named_parameters():
            #print("module name: ",name)
            #print("module size: ",module.size())

            if "dropout" not in name:

                if '.weight' in name:
                    upname=name.replace('.weight','')
                   
                    if upname !='fcout':
                        if 'conv' in upname:
                            mname= 'model.'+upname+'.out_channels'
                            j=eval(mname)
                            mask[upname]=torch.ones(j)
                        elif 'fc' in upname:
                            mname= 'model.'+upname+'.out_features'
                            j=eval(mname)
                            mask[upname]=torch.ones(j)
                    i=i+1
                

    return mask
    

def apply_mask(model, mask):
    with torch.no_grad():
        layer_name=[]
        for name in mask:
            layer_name.append(name)
        layer = 0
        for name, param in model.named_parameters():
            if "dropout" not in name:
                if 'bias' in name:
                    upname=name.replace('.bias','')
                    if upname!='fcout':
                        param.copy_(mask[upname]*param) 
                    layer+=1
                else:
                    upname=name.replace('.weight','')
                    if upname!='fcout':
                        if 'conv' in upname:
                            param.copy_(mask[upname].unsqueeze(1).unsqueeze(2).unsqueeze(3)*param)
                        else:
                            param.copy_(mask[upname].unsqueeze(1)*param)
            
    return model
    
def percent_mask(mask):
    nummask=0
    totalmask=0
    for v in mask.values():
        nummask+=torch.sum(v)
        totalmask += len(v)
    return nummask/totalmask


def update_mask(mask, activation, percentile):
    mean_activation={}
    for i in activation:
        if 'conv' in i:
            t=torch.mean(activation[i].abs(),0)
            #mean_activation[i]=torch.mean(torch.flatten(t))
            mean_activation[i]=torch.mean(torch.flatten(t,1),1)
        else:
            #mean_activation[i]=torch.mean(activation[i].abs())
            mean_activation[i]=torch.mean(activation[i].abs(),0)
    tf=mean_activation[next(iter(mean_activation))]
    i=0
    for key in mean_activation:
        if i!=0:
            tf=torch.hstack((tf,mean_activation[key]))
        i=i+1 
    tf,_=torch.sort(tf)
    tf = tf[tf.nonzero().squeeze()]
    nummask=0
    for v in mask.values():
        nummask+=torch.sum(v)
    threshold=tf[int(percentile*nummask)]
    umask = copy.deepcopy(mask)
    for key in mean_activation:
        for i in range(len(mean_activation[key])):
            if mean_activation[key][i].abs()<threshold:
                umask[key][i]=0
    return umask
    

