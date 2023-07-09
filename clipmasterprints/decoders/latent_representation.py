import torch
# TODO: make a class latent space representation with functions from_image, from_numpy, to_numpy, to_image
class LatentRepresentation:

    def __init__(self, autoencoder):
        self.autoencoder = autoencoder

    def from_image(self,image):
        return self.autoencoder.encode(image)

    def to_image(self, latent_encoding):
        return self.autoencoder.decode(latent_encoding)