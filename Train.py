import torch
from Loss import Loss

def Train(vae, Xn, beta, cond, gen, device, Xc=None):

    tot_epochs = 3
    loss_eval = 500
    batch = 128
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.3, total_iters=len(Xn))

    print('The model has', sum(p.numel() for p in vae.parameters() if p.requires_grad), 'trainable parameters' )

    for epoch in range(tot_epochs):

        running_loss = 0.0
        for i in range(len(Xn)//batch):

            # Batch of training 
            ix = torch.randint(0, len(Xn), (batch,), generator=gen, device=device)

            # Training
            if cond:
                x, y, mu, log_sig = vae(Xn[ix,].view(batch,1,28,28), Xc[ix])
                loss = Loss(x, y, mu, log_sig, beta)
            else:
                x, y, mu, log_sig = vae(Xn[ix,].view(batch,1,28,28))
                loss = Loss(x, y, mu, log_sig, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            scheduler.step()

            if i % loss_eval == loss_eval-1:
                print(f'(epoch: {epoch}), sample: {batch*(i+1)}, ---> train loss = {running_loss/loss_eval:.2f}')
                running_loss = 0.0

    return vae