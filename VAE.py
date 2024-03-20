import torch
from torchvision import transforms
import pandas as pd
import sys

from VAE_models import CVAE, CoCVAE
from Augment import Augm
from Train import Train

def main(beta=1.0, cond=False, hidden_dim=5, fash=False):
        
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print('Device in use:', device)
        
        if fash:
            data1 = pd.read_csv('archive/fashion-mnist_train.csv')
            f_n = 'T'

        else:
            data1 = pd.read_csv('archive/mnist_train.csv')
            f_n = 'F'
        
        data = data1.drop('label', axis = 1)
        d = torch.tensor(data.to_numpy(), device=device)
        
        gen = torch.Generator(device=device)
        gen.manual_seed(123)
        
        if cond:
            Y = torch.tensor(data1['label'].to_numpy(), device='cpu') 
            c_n = 'T'
            X, Xc = Augm(d, 3, cond, device, Y)
            Xn = X/255
            vae = CoCVAE(hidden_dim, gen=gen, device=device).to(device)
            vae = Train(vae, Xn, beta, cond, gen, device, Xc)
        else:
            c_n = 'F'
            X = Augm(d, 3, cond, device)
            Xn = X/255
            vae = CVAE(hidden_dim, gen=gen, device=device).to(device)
            vae = Train(vae, Xn, beta, cond, gen, device)
        
        
        vae_name = f'saved_models/vae_B{beta}_C{c_n}_Z{hidden_dim}_F{f_n}'

        torch.save(vae.state_dict(), vae_name)

if __name__ == "__main__":

    if len(sys.argv) == 2:
         main(beta=int(sys.argv[1]))

    elif len(sys.argv) == 3:
         main(beta=int(sys.argv[1]), cond=bool(int(sys.argv[2])))

    elif len(sys.argv) == 4:
         main(beta=int(sys.argv[1]), cond=bool(int(sys.argv[2])), hidden_dim=int(sys.argv[3]))

    elif len(sys.argv) == 5:
         main(beta=int(sys.argv[1]), cond=bool(int(sys.argv[2])), hidden_dim=int(sys.argv[3]), fash=bool(int(sys.argv[4])))

    else:
         main()    