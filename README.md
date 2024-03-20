The model implemented are VAE, beta-VAE, Conditional VAE. When calling VAE.py, the informartion to insert are, respectively in order:
- Beta (scalar in front of KL loss), which represent shifting from a classical VAE to a Beta-VAE
- Boolean value to set the VAE Conditional (True) or not (False)
- Latent dimension Z
- Boolean value to set the dataset in use: True -> Fashion MNIST, False -> MNIST

The model architectures are in VAE_models.py
The Jupyter notebooks are just stages where new implementations or results were tested
