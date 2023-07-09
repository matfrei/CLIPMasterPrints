import torch
from PIL import Image
import numpy as np
import os

class GradientOptimizer:

    @staticmethod
    def get_default_dict():
        return {'save_after_x_iter': 1000, 'learning_rate':5e-5, 'batch_size':14, 'max_iter':50000}

    # TODO: if this is the same for all classes, we can avoid duplicate code by just deriving all of them from an optimizer interface?
    def construct_hyperparam_dict(self, hyperparams):
        return_dict = GradientOptimizer.get_default_dict()
        if hyperparams is not None:
            for (key, value) in hyperparams.items():
                return_dict[key] = value
        return return_dict

    def __init__(self, loss, latent_dims, device, hyperparams=None, save_callback=None):
        self.loss = loss
        self.hyperparams = self.construct_hyperparam_dict(hyperparams)
        if save_callback is not None:
            self.save_callback = save_callback
        else:
            def save_default(best_image, latents, losses, iter):
                to_np_image = lambda tensor: tensor.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
                best_image = to_np_image(best_image)
                img = Image.fromarray((best_image * 255).astype(np.uint8))
                img.save(os.path.join(f'solution_{self.iter:05}.png'))

            self.save_callback = save_default

        self.learning_rate = self.hyperparams['learning_rate']
        self.batch_size = self.hyperparams['batch_size']
        self.latent_dims = (self.batch_size,) + latent_dims[1:]

        self.max_iter = self.hyperparams['max_iter']
        self.iter = 0
        self.device = device

    def finished(self):
        if self.max_iter is not None:
            return self.iter > self.max_iter
        return True

    def optimize(self):
        latent = torch.rand(self.latent_dims).detach().to(self.device)
        latent.requires_grad = True

        latent_param = torch.nn.parameter.Parameter(data=latent, requires_grad=True)
        optimizer = torch.optim.Adam([latent_param], lr=self.learning_rate)

        while not self.finished():

            images = torch.cat([image[None] for image in self.loss.latent_to_image_tensor(latent_param)])
            loss_all = self.loss.loss_grad(images)
            loss = torch.sum(loss_all)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.iter+=1

            if not (self.iter % 100):
                print(f'Iteration {self.iter}, loss: {loss.item()}, min cand. loss {torch.min(loss_all)}. max cand loss: {torch.max(loss_all)}')

            if not (self.iter % self.hyperparams['save_after_x_iter']):
                best_image = images[torch.argmin(loss_all)][None]
                self.save_callback(best_image, latent, loss_all, self.iter)





