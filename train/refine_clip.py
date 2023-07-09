import torch
import clip
import os
import logging
import pickle

from torch.utils.tensorboard import SummaryWriter

from clipmasterprints import Experiment, ImageFolderCaption, refine_clip_wrong_token_loop, StableDiffusionWrapper,LatentRepresentation, CLIPLoss, CMAESOptimizer

# TODO: make this script configurable using a config file

# set up experiment logging
experiment = Experiment('results-reproduced/CLIPMasterPrints','CLIP-ViT-L14','refine-off-manifold-token-loop-lr-1e-7')
logging.basicConfig(filename=os.path.join(experiment.log_path(), 'train.log'),level=logging.INFO)

# set up tensorboard
writer = SummaryWriter(log_dir=experiment.tb_path())
log_path = os.path.join(experiment.log_path(),'training.log')
logging.basicConfig(filename=log_path,level=logging.INFO)

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

# load clip model
clip_model, preprocess = clip.load('ViT-L/14',device=device,jit=False)
clip_model = clip_model.train()

loss_img = torch.nn.CrossEntropyLoss()
loss_txt = torch.nn.CrossEntropyLoss()

# Ok, now let's try and see what the dataloader makes of this
batch_size = 20
eval_batch_size = 1000
train_set = ImageFolderCaption(root='/home/common/datasets/imagenet2012/train/',mapping_path = 'data/LOC_synset_mapping.txt',transform=preprocess)
val_set = ImageFolderCaption(root='/home/common/datasets/imagenet2012/val/',mapping_path = 'data/LOC_synset_mapping.txt',transform=preprocess)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=eval_batch_size, shuffle=True, num_workers=7, pin_memory=True)

min_loss = 20.

# load the autoencoder model
autoencoder = StableDiffusionWrapper('../external/stable-diffusion/configs/stable-diffusion/v1-inference.yaml', '../external/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt',
                                     image_dims=(224,224)).to(device)
representation = LatentRepresentation(autoencoder)

captions = open('data/imagenet_classes.txt', 'r').read().split('\n')

loss = CLIPLoss(['ViT-L/14'], captions, representation, device=device,
                input_size=224, clip_bs=40,
                rep_bs=15)

flattened_dim = 4 * (224 // 8) * (224 // 8)
optimizer = CMAESOptimizer(loss=loss, n_features=flattened_dim)

# load ES optimizer w found good solution
with open('results/es_imagenet_25.pkl','rb') as pkl_file:
    optimizer.es = pickle.load(pkl_file)

def checkpoint_network(model, epoch, val_loss):
    global min_loss
    if val_loss < min_loss:
        torch.save({'epoch':epoch,'loss':val_loss,'state_dict':model.state_dict()},os.path.join(experiment.weight_path(), 'clip-vit14l_best.pt'))
        min_loss = val_loss
    torch.save({'epoch': epoch, 'loss': val_loss, 'state_dict': model.state_dict()},
               os.path.join(experiment.weight_path(), 'clip-vit14l_latest.pt'))

refine_clip_wrong_token_loop(clip_model, learning_rate=1e-7, num_epochs=10, dataloaders=[train_dataloader,val_dataloader], device=device,tb_writer=writer,cma_optimizer=optimizer,latent_decoder=representation,save_fn=checkpoint_network,log_fn=logging.info)
