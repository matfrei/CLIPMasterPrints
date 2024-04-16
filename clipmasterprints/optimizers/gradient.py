import torch
import torchvision.transforms.v2.functional
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
        self.x_init_path = self.hyperparams['x_init_path']

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
            torch.argmin(loss_all)

            if not (self.iter % 100):
                print(f'Iteration {self.iter}, loss: {loss.item()}, min cand. loss {torch.min(loss_all)}. max cand loss: {torch.max(loss_all)}')

            if not (self.iter % self.hyperparams['save_after_x_iter']):
                best_image = images[torch.argmin(loss_all)][None]
                self.save_callback(best_image, latent, loss_all, self.iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.iter += 1

class RawGradientOptimizer(GradientOptimizer):

    def optimize(self):
        latent = torch.rand(self.latent_dims).detach().to(self.device)
        latent.requires_grad = True

        latent_param = torch.nn.parameter.Parameter(data=latent, requires_grad=True)
        optimizer = torch.optim.Adam([latent_param], lr=self.learning_rate)

        while not self.finished():
            images = torch.clamp(latent_param,min=0,max=1)
            loss_all = self.loss.loss_grad(images)
            loss = torch.sum(loss_all)
            torch.argmin(loss_all)

            if not (self.iter % 100):
                print(
                    f'Iteration {self.iter}, loss: {loss.item()}, min cand. loss {torch.min(loss_all)}. max cand loss: {torch.max(loss_all)}')

            if not (self.iter % self.hyperparams['save_after_x_iter']):
                best_image = images[torch.argmin(loss_all)][None]
                self.save_callback(best_image, latent, loss_all, self.iter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.iter += 1



class PGDOptimizer(GradientOptimizer):

    # Started out here from https://github.com/Harry24k/adversarial - attacks - pytorch
    # Merci for sharing your code!
    def optimize(self):

        original = torchvision.transforms.v2.functional.pil_to_tensor(Image.open(self.x_init_path).convert("RGB"))
        original = original.to(self.device)[None]
        #original = torch.rand(self.latent_dims).detach().to(self.device)
        images = original.detach()
        images.requires_grad = True

        while not self.finished():
            # set all gradients of model parameters None to avoid
            for (model,preprocess) in self.loss.clip_models.values():
                #print(model)
                model.eval()
                model.zero_grad()
            #    for param in model.parameters():
            #        param.grad = None

            loss_all = self.loss.loss_grad(images)
            loss = torch.sum(loss_all)
            loss.backward()

            if not (self.iter % 100):
                print(
                    f'Iteration {self.iter}, loss: {loss.item()}, min cand. loss {torch.min(loss_all)}. max cand loss: {torch.max(loss_all)}')

            if not (self.iter % self.hyperparams['save_after_x_iter']):
                best_image = images[torch.argmin(loss_all)][None]
                self.save_callback(best_image, images, loss_all, self.iter)

            # go a tiny step into lower loss direction
            update_unconstrained = images - self.learning_rate * images.grad.sign()
            # limit the difference between original and updated image, yielding the adverserial additive
            adverserial_additive = torch.clamp(update_unconstrained-original,min=-0.05,max=0.05)
            # add the adverserial additive to the original image, to obtain the adverserial image, and clamp
            images = torch.clamp(original + adverserial_additive, min=0, max=1).detach()
            images.requires_grad = True
            self.iter += 1



class IntPGDOptimizer(GradientOptimizer):

    # Started out here from https://github.com/Harry24k/adversarial - attacks - pytorch
    # Merci for sharing your code!
    def optimize(self):

        original = torchvision.transforms.v2.functional.pil_to_tensor(Image.open(self.x_init_path).convert("RGB"))
        original = original.to(self.device)[None]
        #original = torch.rand(self.latent_dims).detach().to(self.device)
        images = torch.tensor(original,dtype=torch.uint8)

        while not self.finished():
            # set all gradients of model parameters None to avoid
            for (model,preprocess) in self.loss.clip_models.values():
                model.eval()
                model.zero_grad()
            #    for param in model.parameters():
            #        param.grad = None
            images_float = images.detach().float()/255
            images_float.requires_grad = True
            loss_all = self.loss.loss_grad(images_float)
            loss = torch.sum(loss_all)
            loss.backward()

            if not (self.iter % 100):
                print(
                    f'Iteration {self.iter}, loss: {loss.item()}, min cand. loss {torch.min(loss_all)}. max cand loss: {torch.max(loss_all)}')

            if not (self.iter % self.hyperparams['save_after_x_iter']):
                best_image = images[torch.argmin(loss_all)][None]
                self.save_callback(best_image, images, loss_all, self.iter)

            # go a tiny step into lower loss direction
            grad_update = images_float.grad.sign()
            update_unconstrained = images - grad_update
            # limit the difference between original and updated image, yielding the adverserial additive
            adverserial_additive = torch.clamp(update_unconstrained-original,min=-15,max=15)
            # add the adverserial additive to the original image, to obtain the adverserial image, and clamp
            images = torch.tensor(torch.clamp(original + adverserial_additive, min=0, max=255).detach(),dtype=torch.uint8)
            self.iter += 1



