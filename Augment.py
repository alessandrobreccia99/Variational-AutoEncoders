from torchvision import transforms
import torch

def Augm(d, aug_ratio, cond, device, Y=None):

    data_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, scale=(0.95, 1)),
            #transforms.transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            ])
    X = d        
    for a in range(aug_ratio):
            aug = data_transform(d.view(len(d),1,28,28))
            tmp = aug.view(len(d),28*28)
            X = torch.cat((X,tmp), dim=0)
            
    if cond:
        Y = Y.repeat(aug_ratio+1)
        Xc = torch.eye(10)[Y].to(device)
        return X, Xc

    else:
        return X
        