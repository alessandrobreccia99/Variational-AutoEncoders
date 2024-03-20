import torch
import torch.nn as nn

class CVAE(nn.Module):
            def __init__(self, lat, gen, device):
                super().__init__()
                self.encoder = nn.Sequential( 
                 nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), padding=0), nn.MaxPool2d(2,2), nn.Dropout(0.1) , nn.LeakyReLU(),
                 nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,5), padding=0), nn.LeakyReLU(),
                 nn.Flatten()
                 )
                
                self.lat = lat
                self.gen = gen
                self.dev = device
                self.z_mean = nn.Linear(8**2*16, self.lat)
                self.z_log_var = nn.Linear(8**2*16, self.lat)
        
        
                self.decoder = nn.Sequential(nn.Linear(lat,8**2*16), nn.Unflatten(1,(16,8,8)),
                 nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=(5,5), padding=0) , nn.Upsample(scale_factor=2, mode='bilinear') , nn.Dropout(0.1), nn.LeakyReLU(),
                 nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5,5), padding=0),  nn.LeakyReLU(),
                )
            
        
            def forward(self, i):
                
                x = self.encoder(i)
                mu = self.z_mean(x)
                log_sig = self.z_log_var(x)
                z = mu + torch.exp(log_sig/2) * torch.randn((1,self.lat), generator=self.gen, device=self.dev)
                y = self.decoder(z)
                
                return i, y, mu, log_sig
            
            @torch.no_grad()
            def generate(self):
                
                out = self.decoder(torch.randn((1,self.lat), generator=self.gen, device=self.dev))
                return out
            
            @torch.no_grad()
            def decode(self,z):
                
                out = self.decoder(z)
                return out
            
            @torch.no_grad()
            def encode(self,i):
                
                x = self.encoder(i)
                mu = self.z_mean(x)
                log_sig = self.z_log_var(x)
                z = mu + torch.exp(log_sig/2) *  torch.randn((1,self.lat), generator=self.gen, device=self.dev)
                return z
            
            @torch.no_grad()
            def reproduce(self, i):
                
                x = self.encoder(i)
                mu = self.z_mean(x)
                log_sig = self.z_log_var(x)
                z = mu + torch.exp(log_sig/2) * torch.randn((1,self.lat), device=self.dev)
                y = self.decoder(z)
        
                return y
            
class CoCVAE(nn.Module):
    def __init__(self, lat, gen, device):
        super().__init__()
        self.encoder = nn.Sequential( 
         nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5), padding=0), nn.MaxPool2d(2,2), nn.Dropout(0.1) , nn.LeakyReLU(),
         nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,5), padding=0), nn.LeakyReLU(),
         nn.Flatten()
         )
        
        self.lat = lat
        self.gen = gen
        self.dev = device
        self.z_mean = nn.Linear(8**2*16, self.lat)
        self.z_log_var = nn.Linear(8**2*16, self.lat)

        self.decoder = nn.Sequential(nn.Linear(lat+10,8**2*16), nn.Unflatten(1,(16,8,8)),
         nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=(5,5), padding=0) , nn.Upsample(scale_factor=2, mode='bilinear') , nn.Dropout(0.1), nn.LeakyReLU(),
         nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5,5), padding=0),  nn.LeakyReLU(),
        )
    

    def forward(self, i, c):
        
        x = self.encoder(i)
        mu = self.z_mean(x)
        log_sig = self.z_log_var(x)
        z = mu + torch.exp(log_sig/2) * torch.randn((1,self.lat), generator=self.gen, device=self.dev) 
        y = self.decoder(torch.cat((z,c), dim=1))
    
        return i, y, mu, log_sig
    
    @torch.no_grad()
    def generate(self,c):
        
        z = torch.randn((1,self.lat), generator=self.gen, device=self.dev)
        out = self.decoder(torch.cat((z,c), dim=1))
        return out
    
    @torch.no_grad()
    def decode(self,z,c):

        out = self.decoder(torch.cat((z,c), dim=1))
        return out
    
    @torch.no_grad()
    def encode(self,i):
        
        x = self.encoder(i)
        mu = self.z_mean(x)
        log_sig = self.z_log_var(x)
        z = mu + torch.exp(log_sig/2) * torch.randn((1,self.lat), generator=self.gen, device=self.dev)
        return z
    
    @torch.no_grad()
    def reproduce(self, i, c):
        
        x = self.encoder(i)
        mu = self.z_mean(x)
        log_sig = self.z_log_var(x)
        z = mu + torch.exp(log_sig/2) * torch.randn((1,self.lat), generator=self.gen, device=self.dev)
        y = self.decoder(z)

        return y