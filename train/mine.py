import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

import clip
import torch,torchvision
import argparse
from pytorch_lightning import seed_everything
from config.defaults import get_cfg_defaults
from PIL import Image
from pathlib import Path

from clipmasterprints import Experiment, StableDiffusionWrapperWGradient, LatentRepresentation, CLIPLoss, ShiftedCLIPLoss, CMAESOptimizer, GradientOptimizer,clip_extract_image_embeddings_on_demand,build_clip
from functools import partial

candidate_to_2d = lambda input, shape: input.reshape(shape).astype('float32')
to_np_image = lambda tensor: tensor.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

def save_solution(es_object, iter, outpath, weightpath, latent_shape, representation, torch_device):
    with open(os.path.join(weightpath, f'es.pkl'), 'wb') as file:
        pickle.dump(es_object, file)
        print(es_object.result)

    best = torch.from_numpy(candidate_to_2d(es_object.result.xbest, latent_shape)).to(torch_device)
    favorite = torch.from_numpy(candidate_to_2d(es_object.result.xfavorite, latent_shape)).to(torch_device)

    best_image = to_np_image(representation.to_image(best))
    favorite_image = to_np_image(representation.to_image(favorite))

    plt.imshow(best_image)
    plt.show()

    plt.imshow(favorite_image)
    plt.show()

    img = Image.fromarray((best_image * 255).astype(np.uint8))
    img.save(os.path.join(outpath, f'solution_{iter:05}.png'))

    img = Image.fromarray((favorite_image * 255).astype(np.uint8))
    img.save(os.path.join(outpath, f'solution_favorite{iter:05}.png'))

def save_solution_grad(best_image,latents,losses, iter, outpath):
    best_image = to_np_image(best_image)
    plt.imshow(best_image)
    plt.show()
    img = Image.fromarray((best_image * 255).astype(np.uint8))
    img.save(os.path.join(outpath,f'solution_{iter:05}.png'))

def parse_args():
    parser = argparse.ArgumentParser(
        prog='mine',
        description='Search for fooling master images (CLIPMasterPrints)',
        epilog='TODO: Add text at the bottom of help')
    parser.add_argument('--config-path', help='path to config file')
    args = parser.parse_args()
    return args

def get_optimizer_class(str_conf):
    if str_conf == "SGD":
        optimizer_class = GradientOptimizer
    else:
        # default optimizer is always CMA-ES
        optimizer_class = CMAESOptimizer
        # we won't need any gradients in this case, swtich them off
        torch.set_grad_enabled(False)
    return optimizer_class

def find_fooling():
    config_path = parse_args().config_path
    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    config.freeze()

    print(config_path)
    print(config)

    # use gpu as device if available
    device = torch.device(f'cuda:{config.GPU_ID}') if torch.cuda.is_available() else torch.device("cpu")

    # set up experiment logging
    experiment = Experiment(config.EXPERIMENT_LOG.BASEPATH, config.EXPERIMENT_LOG.MODEL_NAME, config.EXPERIMENT_LOG.EXPERIMENT_NAME)
    logging.basicConfig(filename=os.path.join(experiment.log_path(),'mine.log'))

    # load the autoencoder model
    autoencoder = StableDiffusionWrapperWGradient(config.AUTOENCODER.CONFIG_PATH, config.AUTOENCODER.WEIGHT_PATH, image_dims=(config.AUTOENCODER.IMG_HEIGHT,config.AUTOENCODER.IMG_WIDTH)).to(device)
    # TODO: After refactoring, LatentRepresentation is unnecessary. Adjust interfaces!
    representation = LatentRepresentation(autoencoder)

    captions = open(config.DATA.CAPTION_PATH,'r').read().split('\n')
    # filter empty strings
    captions = [caption for caption in captions if caption]
    ac_dims = (1,config.AUTOENCODER.LATENT_CHANNELS,config.AUTOENCODER.IMG_HEIGHT // config.AUTOENCODER.DOWNSAMPLING_FACTOR,config.AUTOENCODER.IMG_WIDTH // config.AUTOENCODER.DOWNSAMPLING_FACTOR)

    # arrange hyperparams into dict
    hyperparam_dict = {'save_after_x_iter': config.OPTIMIZER.CHECK_POINT_AFTER_X_ITER, 'init_vector':'norm', 'sigma_0': 1., 'pop_size':'default', 'max_iter': config.OPTIMIZER.ITER, 'learning_rate': config.OPTIMIZER.LR, 'batch_size': config.OPTIMIZER.BATCH_SIZE}
    num_runs = config.NUM_RUNS
    sample_from_captions_lst = config.SAMPLE_CAPTIONS

    for sample_from_captions in sample_from_captions_lst:
        for run_idx in range(num_runs):
            # seed random generator
            seed_everything(config.RANDOM_SEED + run_idx)

            outpath = os.path.join(experiment.output_path(),f'run{run_idx}_sample_{sample_from_captions}')
            weightpath = os.path.join(experiment.weight_path(),f'run{run_idx}_sample_{sample_from_captions}')
            Path(outpath).mkdir(parents=True, exist_ok=True)
            Path(weightpath).mkdir(parents=True, exist_ok=True)

            if sample_from_captions != 'all':
                caption_indices = np.random.permutation(len(captions))[:sample_from_captions].tolist()
            else:
                caption_indices = list(range(len(captions)))

            current_captions = [captions[idx] for idx in caption_indices]

            # log captions such that we know later which ones where used in training
            with open(os.path.join(outpath,'optimized_captions.txt'),'w') as outfile:
                for caption in current_captions:
                    outfile.write(caption)
                    outfile.write('\n')

            clip_models = dict([(clip_string, (clip_model, preprocessing)) for (clip_string, clip_model, preprocessing) in
                    [build_clip(clip_string,device=device) for clip_string in config.CLIP.MODEL_STRINGS]])

            # TODO: FIXXXME: spaghetti code, wrap into function or class
            if not np.isclose(config.CLIP.SHIFT,0):

                tokens = clip.tokenize(captions).to(device)
                train_set = torchvision.datasets.ImageFolder(root=config.DATA.IMAGENET_TRAIN)
                train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.DATA.BS_IMAGENET_EVAL,
                                                           num_workers=7)
                gap_vectors = {}
                for (key, (model, preprocessing)) in clip_models.items():
                    # FIXXXME don't load model twice just to get preprocessing w resize
                    _,preprocess = clip.load(key,device='cpu')
                    train_set.transform = preprocess
                    feature_path = f'{config.FEATURES.PATH_TRAIN[:-3]}_{key.replace("/","-")}.pt'
                    print(feature_path)
                    (train_embeddings,train_labels) = clip_extract_image_embeddings_on_demand(model, train_loader, feature_path,device=device)
                    image_features_mean = torch.mean(train_embeddings,dim=0,keepdim=True)
                    with torch.no_grad():
                        text_features = model.encode_text(tokens)
                    text_features_mean = torch.mean(text_features)
                    gap_vectors[key] = config.CLIP.SHIFT * (image_features_mean - text_features_mean)

                loss = ShiftedCLIPLoss(clip_models, current_captions, representation, device=device, input_size=config.AUTOENCODER.IMG_HEIGHT, clip_bs=config.CLIP.BATCH_SIZE, rep_bs=config.AUTOENCODER.BATCH_SIZE,
                                       gap_vectors=gap_vectors)
            else:
                loss = CLIPLoss(clip_models, current_captions, representation, device=device, input_size=config.AUTOENCODER.IMG_HEIGHT, clip_bs=config.CLIP.BATCH_SIZE, rep_bs=config.AUTOENCODER.BATCH_SIZE)
            #FIXXME: this is hacky, find better solution to unify callback signatures
            if config.OPTIMIZER.METHOD == 'SGD':
                checkpoint_fn = partial(save_solution_grad, outpath=outpath)
            else:
                checkpoint_fn = partial(save_solution, outpath=outpath, weightpath=weightpath,
                                    latent_shape=(1,) + loss.latent_shape[1:], representation=representation,
                                    torch_device=device)

            optimizer = get_optimizer_class(config.OPTIMIZER.METHOD)(loss=loss, latent_dims=ac_dims, hyperparams=hyperparam_dict, device=device,save_callback=checkpoint_fn)
            optimizer.optimize()

if __name__ == '__main__':
    find_fooling()
