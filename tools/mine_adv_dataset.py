import os
import tqdm
import numpy as np
import torch, torchvision
from clipmasterprints import IntPGDOptimizer, CLIPLoss, IdentityRepresentation, build_clip
from pytorch_lightning import seed_everything
from PIL import Image
from functools import partial

to_np_image = lambda tensor: tensor.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
def save_solution_grad_uint8(best_image,latents,losses, iter, outpath,batch_idx):
    if iter == 0:
        return
    tensor = best_image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    img = Image.fromarray(tensor.astype(np.uint8))
    img.save(os.path.join(outpath,f'image_{idx}.png'))

# paths
imagenet_root = '~/data/imagenet2012/train'

## params

# batch size
batch_size = 496
# device
device = torch.device("cuda:0")
# which clip model
clip_string = 'ViT-B/32'

# input_size
input_size = 224
# representation
latent_dims = (1, 3, input_size,input_size)

# load clip model
(clip_string,clip_model,preprocessing) = build_clip(clip_string, device=device)
clip_models = dict([(clip_string, (clip_model, preprocessing))])

# captions
sample_from_captions = 25
caption_path = 'data/imagenet_classes.txt'
captions = open(caption_path, 'r').read().split('\n')

# filter empty strings
captions = [caption for caption in captions if caption]

representation = IdentityRepresentation()

train_set = torchvision.datasets.ImageFolder(root=imagenet_root)

outpath = "~/data/adv_imagenet2012/train/tainted"

for idx,(filepath,label) in enumerate(tqdm.tqdm(train_set.samples)):
    seed_everything(idx)
    caption_indices = np.random.permutation(len(captions))[:sample_from_captions].tolist()
    current_captions = [captions[idx] for idx in caption_indices]
    loss = CLIPLoss(clip_models, current_captions, representation, device=device, input_size=input_size,
                    clip_bs=batch_size, rep_bs=None)
    checkpoint_fn = partial(save_solution_grad_uint8, outpath=outpath, batch_idx=idx)
    hyperparam_dict = {'save_after_x_iter': 100, 'max_iter': 100, 'batch_size': batch_size, 'x_init_path': filepath}
    optimizer = IntPGDOptimizer(loss=loss, latent_dims=latent_dims, hyperparams=hyperparam_dict, device=device, save_callback=checkpoint_fn)
    optimizer.optimize()
