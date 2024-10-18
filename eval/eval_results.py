import os
import json

from PIL import Image
import torch, torchvision
import clip
from pytorch_lightning import seed_everything
import numpy as np
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from open_clip import get_tokenizer
from clipmasterprints import clip_extract_image_embeddings_on_demand, clip_extract_image_embeddings, blip_extract_image_embeddings, siglip_extract_image_embeddings, build_blip, build_siglip, get_similarities_per_class, scatter_optimized_classes, eval_fooling_accuracy, scatter_optimized_classes_multi,plot_similarity_heatmap

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# TODO: adjust paths to match locations of imagenet train and val set on your system
IMAGENET_TRAIN_PATH = '~/data/imagenet2012/train'
IMAGENET_VAL_PATH = '~/data/imagenet2012/val'

# TODO: adjust eval batch size as fitting for your setup
EVAL_BATCH_SIZE = 100
NUM_WORKERS = 2

FIGURE_OUTPUT_PATH = f"figures/"
Path(FIGURE_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

def plot_art_heatmap():
    clip_model, preprocess = clip.load('ViT-L/14@336px',device=device)
    clip_model.eval()

    captions = open('data/famous_paintings.txt').read().split('\n') + ['Random noise image']
    img_filenames = ['Mona_Lisa.jpeg','last_supper.jpeg','starry_night.jpeg','scream.jpeg','Gernika.jpeg','the_kiss.jpeg','girl_with_a_pearl_earring.jpg','birth_of_venus.jpeg','las_meninas.jpeg','creation_of_adam.jpeg',]
    img_paths = [os.path.join('data/paintings',filename) for filename in img_filenames]
    img_paths += ['results/baseline/random.png','results/master_images/cmp_artworks_pgd_int.png']
    images = [Image.open(path).convert("RGB") for path in img_paths]
    images_pp = [preprocess(image) for image in images]

    image_input = torch.tensor(np.stack(images_pp)).to(device)
    tokens_input = clip.tokenize(captions).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(tokens_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity_art = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    np.savetxt('figure1_data.txt',similarity_art)
    plot_similarity_heatmap(captions, images, similarity_art,os.path.join(FIGURE_OUTPUT_PATH,'figure_1.pdf'))


def sample_opt_idcs(num_captions,random_seed = 0):
    # get optimized captions
    seed_everything(random_seed)
    return np.random.permutation(len(captions))[:num_captions].tolist()

def sample_opt_captions(num_captions,random_seed = 0):
    # get optimized captions
    caption_indices = sample_opt_idcs(num_captions,random_seed)
    return [captions[idx] for idx in caption_indices]

def set_up_imagenet_features(clip_model,preprocess,train=False):
    # TODO: update path to correct imagenet validation set location here

    if train:
        imagenet_path = IMAGENET_TRAIN_PATH
        features_filename = 'features/imagenet_train_clip_embeddings.pt'
        dataset_name = 'train'
    else:
        imagenet_path = IMAGENET_VAL_PATH
        features_filename = 'features/imagenet_val_clip_embeddings.pt'
        dataset_name = 'validation'

    print(f'Loading CLIP embeddings for ImageNet {dataset_name} set')
    print(f'This could take a while ...')


    mapping_path = 'data/LOC_synset_mapping.txt'
    mapping_lst = open(mapping_path, 'r').read().split('\n')
    mapping = dict([(string_pair[:9], string_pair[9:].strip()) for string_pair in mapping_lst if string_pair])
    # get ImageNet validation set
    data_set = torchvision.datasets.ImageFolder(root=imagenet_path, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=EVAL_BATCH_SIZE, num_workers=NUM_WORKERS)

    idx_to_class = OrderedDict([(value, mapping[key]) for key, value in data_set.class_to_idx.items()])
    captions = list(idx_to_class.values())

    (features_unnorm, labels) = clip_extract_image_embeddings_on_demand(clip_model, data_loader, features_filename, device=device)
    val_features = features_unnorm / features_unnorm.norm(dim=-1, keepdim=True)
    return val_features, labels, captions, idx_to_class, features_unnorm

def set_up_imagenet_features_custom(clip_model_str,train=False, model_dict=None):
    # TODO: update path to correct imagenet validation set location here
    if clip_model_str == "BLIP-384":
        extractor_callback = blip_extract_image_embeddings
        # create BLIP model here
        (_,clip_model,preprocess) = build_blip(clip_model_str,device,tensor_input=False)

    elif clip_model_str == "ViT-L-16-SigLIP-384":
        extractor_callback = siglip_extract_image_embeddings
        (_, clip_model, preprocess) = build_siglip(clip_model_str, device, tensor_input=False)
    else:
        extractor_callback = clip_extract_image_embeddings
        clip_model, preprocess = clip.load(clip_model_str, device=device)
    clip_model.eval()

    if train:
        imagenet_path = IMAGENET_TRAIN_PATH
        features_filename = f"features/imagenet_train_clip_embeddings_{clip_model_str.replace('/','_')}.pt"
        dataset_name = 'train'
    else:
        imagenet_path = IMAGENET_VAL_PATH
        features_filename = f"features/imagenet_val_clip_embeddings_{clip_model_str.replace('/','_')}.pt"
        dataset_name = 'validation'

    print(f'Loading CLIP embeddings for ImageNet {dataset_name} set')
    print(f'This could take a while ...')

    mapping_path = 'data/LOC_synset_mapping.txt'
    mapping_lst = open(mapping_path, 'r').read().split('\n')
    mapping = dict([(string_pair[:9], string_pair[9:].strip()) for string_pair in mapping_lst if string_pair])
    # get ImageNet validation set
    data_set = torchvision.datasets.ImageFolder(root=imagenet_path, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=EVAL_BATCH_SIZE, num_workers=NUM_WORKERS)

    idx_to_class = OrderedDict([(value, mapping[key]) for key, value in data_set.class_to_idx.items()])
    captions = list(idx_to_class.values())

    (features_unnorm, labels) = clip_extract_image_embeddings_on_demand(clip_model, data_loader, features_filename, device=device, extractor_callback=extractor_callback)

    val_features = features_unnorm / features_unnorm.norm(dim=-1, keepdim=True)
    if model_dict is not None:
        model_dict[clip_model_str]=(clip_model,preprocess)
    return clip_model, val_features, labels, captions, idx_to_class, features_unnorm

def get_imagenet_similarities(clip_model,val_features, val_labels, captions, idx_to_class):

    tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities_imagenet = get_similarities_per_class(val_features, val_labels, text_features, idx_to_class)
    similarities_imagenet = dict(
        (key, [item for sublist in [element.cpu().numpy().tolist()[0] for element in value] for item in sublist]) for
        key, value in similarities_imagenet.items())
    return similarities_imagenet

def get_imagenet_similarities_blip(blip_model, val_features, val_labels, captions, idx_to_class):

    tokens = blip_model.tokenizer(captions, padding='max_length', truncation=True, max_length=35,
                                  return_tensors="pt").to(device)
    with torch.no_grad():
        text_output = blip_model.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask,
                                          return_dict=True, mode='text')
        text_features = F.normalize(blip_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

    similarities_imagenet = get_similarities_per_class(val_features, val_labels, text_features, idx_to_class)
    similarities_imagenet = dict(
        (key, [item for sublist in [element.cpu().numpy().tolist()[0] for element in value] for item in sublist]) for
        key, value in similarities_imagenet.items())
    return similarities_imagenet

def get_imagenet_similarities_siglip(siglip_model, val_features, val_labels, captions, idx_to_class):

    tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-384')
    tokens = tokenizer(captions, context_length=siglip_model.context_length).to(device)
    with torch.no_grad():
        text_features = siglip_model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities_imagenet = get_similarities_per_class(val_features, val_labels, text_features, idx_to_class, siglip=True, logit_scale=siglip_model.logit_scale,
                                                       logit_bias=siglip_model.logit_bias)
    similarities_imagenet = dict(
        (key, [item for sublist in [element.cpu().numpy().tolist()[0] for element in value] for item in sublist]) for
        key, value in similarities_imagenet.items())
    return similarities_imagenet

def plot_imagenet_scatter(clip_model, preprocess, similarities_imagenet,opt_captions):

    # now get scores for fooling image
    fooling_path_cma = 'results/master_images/cmp_imagenet_cma_25.png'
    fooling_path_sgd = 'results/master_images/cmp_imagenet_sgd_25.png'
    fooling_path_pgd = 'results/master_images/cmp_imagenet_pgd_int_25.png'


    opt_tokens = clip.tokenize(opt_captions).to(device)
    img_paths = [fooling_path_cma,fooling_path_sgd,fooling_path_pgd]
    images = [Image.open(path).convert("RGB") for path in img_paths]
    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(opt_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities_fooling = image_features @ text_features.T
    similarities_imagenet_opt = dict(
        [(key, value) for key, value in similarities_imagenet.items() if key in opt_captions])
    scatter_optimized_classes_multi(opt_captions,[('LVE',similarities_fooling[0,:][None].cpu().numpy()),('SGD',similarities_fooling[1,:][None].cpu().numpy()),('PGD',similarities_fooling[2,:][None].cpu().numpy())], similarities_imagenet_opt,os.path.join(FIGURE_OUTPUT_PATH,'figure_3_a.pdf'))

def plot_lines_multi(similarities_imagenet):
    samples = [25, 50, 75, 100]

    fontsize = 10
    params = {  # 'backend': 'pdf',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        # 'text.usetex': True
        # 'text.latex.preamble':r'\usepackage{sfmath}', # \boldmath',
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial'}
    plt.rcParams.update(params)
    clip_scores_optimized = []
    clip_scores_imagenet = []
    average_clip_scores_all = []
    paths = {
        25: [('LVE','results/master_images/cmp_imagenet_cma_25.png'),('SGD','results/master_images/cmp_imagenet_sgd_25.png'),('PGD','results/master_images/cmp_imagenet_pgd_int_25.png')],
        50: [('LVE','results/master_images/cmp_imagenet_cma_50.png'),('SGD','results/master_images/cmp_imagenet_sgd_50.png'),('PGD','results/master_images/cmp_imagenet_pgd_int_50.png')],
        75: [('LVE','results/master_images/cmp_imagenet_cma_75.png'),('SGD','results/master_images/cmp_imagenet_sgd_75.png'),('PGD','results/master_images/cmp_imagenet_pgd_int_75.png')],
        100: [('LVE','results/master_images/cmp_imagenet_cma_100.png'),('SGD','results/master_images/cmp_imagenet_sgd_100.png'),('PGD','results/master_images/cmp_imagenet_pgd_int_100.png')]}  # },/solution_favorite50000.png'}

    for sample in samples:
        # path = f'./run0_sample_{sample}'
        opt_captions = sample_opt_captions(sample)
        opt_captions = [caption for caption in opt_captions if caption]
        tokens_input = clip.tokenize(opt_captions).to(device)
        with torch.no_grad():
            text_features = clip_model_vit.encode_text(tokens_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        clip_scores_imagenet.extend(
            [(key, value, sample, 'ImageNet') for key, tensor_lst in similarities_imagenet.items() for value in tensor_lst if
             key in opt_captions])

        for meth_idx in range(len(paths[sample])):
            img_paths = [paths[sample][meth_idx][1]]
            print(img_paths)
            images = [Image.open(path).convert("RGB") for path in img_paths]
            images = [preprocess_vit(image) for image in images]
            image_input = torch.tensor(np.stack(images)).to(device)

            with torch.no_grad():
                image_features = clip_model_vit.encode_image(image_input).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity_single = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            clip_scores_optimized.extend([(key, value[0], sample, f'{paths[sample][meth_idx][0]}') for key, value in zip(opt_captions, similarity_single.tolist())])

        # TODO: this needs to be transformed, we should only compute ImageNet similarities on demand once


    input_data = clip_scores_imagenet + clip_scores_optimized
    df = pd.DataFrame(input_data, columns=['class', 'mean CLIP score', 'number of optimized classes', 'type'])
    plt.figure()
    sns_plot = sns.catplot(
        data=df, kind="point",
        x='number of optimized classes', y="mean CLIP score", hue='type', dodge=True, height=2.26, aspect=1.416,
        palette=sns.color_palette('colorblind', n_colors=len(paths)+1), legend=False)

    axes = sns_plot.axes.flatten()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    sns_plot.savefig(os.path.join(FIGURE_OUTPUT_PATH,'figure_3_b.pdf'), dpi=300)

def eval_clip(images, captions,clip_string):

    (model, preprocess) = model_dict[clip_string]
    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)
    if clip_string == 'BLIP-384':
        tokens = model.tokenizer(captions, padding='max_length', truncation=True, max_length=35,
                                      return_tensors="pt").to(device)
        with torch.no_grad():
            text_output = model.text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask,
                                                  return_dict=True, mode='text')
            text_features = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)
            image_embeds = model.visual_encoder(image_input)
            image_features = F.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1)
            sims = torch.mm(text_features, image_features.T)
    else:
        if clip_string == 'ViT-L-16-SigLIP-384':
            tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-384')
            tokens_input = tokenizer(captions, context_length=model.context_length).to(device)
        else:
            tokens_input = clip.tokenize(captions).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(tokens_input).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        sims = torch.mm(text_features, image_features.T)
        if clip_string == 'ViT-L-16-SigLIP-384':
            with torch.no_grad():
                sims = torch.sigmoid(sims * model.logit_scale.exp() + model.logit_bias)
    return sims.cpu().numpy()

def plot_lines_multi_other_approaches(similarities_imagenet_vit, similarities_imagenet_resnet, similarities_imagenet_blip, similarities_imagenet_siglip):
    samples = [25, 50, 75, 100]

    fontsize = 10
    params = { 
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial'}
    plt.rcParams.update(params)
    clip_scores_optimized = []
    clip_scores_imagenet = []
    average_clip_scores_all = []

    paths = {
        25: [('PGD_ViT-L/14','results/master_images/cmp_imagenet_pgd_int_25.png'),('PGD_RN50x64','results/master_images/cmp_imagenet_pgd_int_25_rn50x64.jpg'),('PGD_BLIP-384','results/master_images/cmp_imagenet_pgd_int_25_blip.jpg'),('PGD_ViT-L-16-SigLIP-384','results/master_images/cmp_imagenet_pgd_int_25_siglip.jpg')],
        50: [('PGD_ViT-L/14','results/master_images/cmp_imagenet_pgd_int_50.png'),('PGD_RN50x64','results/master_images/cmp_imagenet_pgd_int_50_rn50x64.jpg'),('PGD_BLIP-384','results/master_images/cmp_imagenet_pgd_int_25_blip.jpg'),('PGD_ViT-L-16-SigLIP-384','results/master_images/cmp_imagenet_pgd_int_50_siglip.jpg')],
        75: [('PGD_ViT-L/14','results/master_images/cmp_imagenet_pgd_int_75.png'),('PGD_RN50x64','results/master_images/cmp_imagenet_pgd_int_75_rn50x64.jpg'),('PGD_BLIP-384','results/master_images/cmp_imagenet_pgd_int_25_blip.jpg'),('PGD_ViT-L-16-SigLIP-384','results/master_images/cmp_imagenet_pgd_int_75_siglip.jpg')],
        100: [('PGD_ViT-L/14','results/master_images/cmp_imagenet_pgd_int_100.png'),('PGD_RN50x64','results/master_images/cmp_imagenet_pgd_int_100_rn50x64.jpg'),('PGD_BLIP-384','results/master_images/cmp_imagenet_pgd_int_25_blip.jpg'),('PGD_ViT-L-16-SigLIP-384','results/master_images/cmp_imagenet_pgd_int_100_siglip.jpg')]} 

    for sample in samples:
        
        opt_captions = sample_opt_captions(sample)
        opt_captions = [caption for caption in opt_captions if caption]

        clip_scores_imagenet.extend(
            [(key, value, sample, 'ImageNet_ViT-L/14') for key, tensor_lst in similarities_imagenet_vit.items() for value in tensor_lst if
             key in opt_captions])
        clip_scores_imagenet.extend(
            [(key, value, sample, 'ImageNet_RN50x64') for key, tensor_lst in similarities_imagenet_resnet.items() for value in
             tensor_lst if
             key in opt_captions])
        clip_scores_imagenet.extend(
            [(key, value, sample, 'ImageNet_BLIP-384') for key, tensor_lst in similarities_imagenet_blip.items() for value
             in
             tensor_lst if
             key in opt_captions])
        clip_scores_imagenet.extend(
            [(key, value, sample, 'ImageNet_ViT-L-16-SigLIP-384') for key, tensor_lst in similarities_imagenet_siglip.items() for value
             in
             tensor_lst if
             key in opt_captions])

        for meth_idx in range(len(paths[sample])):
            img_paths = [paths[sample][meth_idx][1]]
            print(img_paths)
            images = [Image.open(path).convert("RGB") for path in img_paths]
            similarity_single = eval_clip(images,opt_captions,paths[sample][meth_idx][0].split('_')[1])
            clip_scores_optimized.extend([(key, value[0], sample, f'{paths[sample][meth_idx][0]}') for key, value in zip(opt_captions, similarity_single.tolist())])

        # TODO: this needs to be transformed, we should only compute ImageNet similarities on demand once
    input_data = clip_scores_imagenet + clip_scores_optimized
    df = pd.DataFrame(input_data, columns=['class', 'mean CLIP score', 'number of optimized classes', 'type'])
    plt.figure()
    sns_plot = sns.catplot(
        data=df, kind="point",
        x='number of optimized classes', y="mean CLIP score", hue='type', dodge=False, height=2*2.26, aspect=1.416,
        palette=sns.color_palette('colorblind', n_colors=2*len(paths)), legend=False)

    axes = sns_plot.axes.flatten()
    plt.legend(loc='upper right',bbox_to_anchor=(1.,.95))
    plt.tight_layout()
    plt.show()
    sns_plot.savefig(os.path.join(FIGURE_OUTPUT_PATH,'figure_5.pdf'), dpi=300)


def generate_poi_table(val_features_unnorm, val_lables, captions, clip_model, preprocess):

    caption_indices = sample_opt_idcs(25)
    other_indices = [idx for idx in range(len(captions)) if not idx in caption_indices]
    sgd_img_path = 'results/master_images/cmp_imagenet_sgd_25.png'
    lve_img_path = 'results/master_images/cmp_imagenet_cma_25.png'
    pgd_img_path = 'results/master_images/cmp_imagenet_pgd_int_25.png'
    sgd_shift_img_path = 'results/master_images/cmp_imagenet_sgd_25_shift_0.25.png'
    lve_shift_img_path = 'results/master_images/cmp_imagenet_cma_25_shift_0.25.png'
    pgd_shift_img_path = 'results/master_images/cmp_imagenet_pgd_int_25_shift_0.25.png'
    paths = [sgd_img_path,lve_img_path,pgd_img_path,sgd_shift_img_path, lve_shift_img_path, pgd_shift_img_path]

    images = [Image.open(path).convert("RGB") for path in paths]
    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)

    tokens = clip.tokenize(captions).to(device)
    with torch.no_grad():
        adv_features_unnorm = clip_model.encode_image(image_input).float()
        text_features_unnorm = clip_model.encode_text(tokens).float()
    val_features_unnorm = val_features_unnorm.float()
    val_features = val_features_unnorm/val_features_unnorm.norm(dim=-1, keepdim=True)
    adv_features = adv_features_unnorm/adv_features_unnorm.norm(dim=-1, keepdim=True)
    text_features = text_features_unnorm/text_features_unnorm.norm(dim=-1, keepdim=True)


    poi_sgd = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[0,:][None], caption_indices)
    poi_lve = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[1,:][None], caption_indices)
    poi_pgd = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[2,:][None], caption_indices)
    poi_sgd_shift = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[3,:][None], caption_indices)
    poi_lve_shift = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[4,:][None], caption_indices)
    poi_pgd_shift = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[5,:][None], caption_indices)

    print(poi_lve)
    print("without test time shift:")
    print(f"SGD, no training shift : {poi_sgd:0.2f}%")
    print(f"LVE, no training shift : {poi_lve:0.2f}%")
    print(f"PGD, no training shift : {poi_pgd:0.2f}%")
    print(f"SGD, with training shift : {poi_sgd_shift:0.2f}%")
    print(f"LVE, with training shift : {poi_lve_shift:0.2f}%")
    print(f"PGD, with training shift : {poi_pgd_shift:0.2f}%")

    _, _, _, _, train_features_unnorm = set_up_imagenet_features(clip_model, preprocess, train=True)

    # compute mean vector for gap shift
    image_features_mean = torch.mean(train_features_unnorm, dim=0, keepdim=True)
    text_features_mean = torch.mean(text_features_unnorm, dim=0, keepdim=True)
    # now define gap vector
    gap_vector = image_features_mean - text_features_mean
    gap_shift = 0.25
    val_features_shift_un = val_features_unnorm - gap_shift * gap_vector
    adv_features_shift_un = adv_features_unnorm - gap_shift * gap_vector
    text_features_shift_un = text_features_unnorm + gap_shift * gap_vector

    val_features_shift = val_features_shift_un/val_features_shift_un.norm(dim=-1, keepdim=True)
    adv_features_shift = adv_features_shift_un/adv_features_shift_un.norm(dim=-1, keepdim=True)
    text_features_shift = text_features_shift_un /text_features_shift_un.norm(dim=-1, keepdim=True)

    poi_sgd_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[0,:][None], caption_indices)
    poi_lve_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[1,:][None], caption_indices)
    poi_pgd_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[2,:][None], caption_indices)
    poi_sgd_shift_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[3,:][None], caption_indices)
    poi_lve_shift_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[4,:][None], caption_indices)
    poi_pgd_shift_1 = 100*eval_fooling_accuracy(val_features_shift, val_labels, text_features_shift, adv_features_shift[5,:][None], caption_indices)

    print("_with_ test time shift:")
    print(f"SGD, no training shift : {poi_sgd_1:0.2f}%")
    print(f"LVE, no training shift : {poi_lve_1:0.2f}%")
    print(f"PGD, no training shift : {poi_pgd_1:0.2f}%")
    print(f"SGD, with training shift : {poi_sgd_shift_1:0.2f}%")
    print(f"LVE, with training shift : {poi_lve_shift_1:0.2f}%")
    print(f"PGD, with training shift : {poi_pgd_shift_1:0.2f}%")

    poi_sgd_other = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[0,:][None], other_indices)
    poi_lve_other = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[1,:][None], other_indices)
    poi_pgd_other = 100*eval_fooling_accuracy(val_features.float(), val_labels, text_features, adv_features[2,:][None], other_indices)

    print("performance on classes not targeted during optimization")
    print(f"SGD, no training shift : {poi_sgd_other:0.2f}%")
    print(f"LVE, no training shift : {poi_lve_other:0.2f}%")
    print(f"PGD, no training shift : {poi_pgd_other:0.2f}%")

def compare_images(img_path, adv_path, clip_model, preprocess):
    images = [Image.open(path).convert("RGB") for path in [img_path,adv_path]]
    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = image_features[0,:][None] @ image_features[1,:][None].T
    print(similarity)

def wordnet_transfer():
    filepaths = dict([('dog', 'results/master_images/wordnet/cmp_wordnet_pgd_dog.png'),
                      ('vegetable', 'results/master_images/wordnet/cmp_wordnet_pgd_vegetable.png'),
                      ('motor vehicle', 'results/master_images/wordnet/cmp_wordnet_pgd_motor_vehicle.png'),
                      ('musical instrument', 'results/master_images/wordnet/cmp_wordnet_pgd_musical_instrument.png')])

    hyponym = {}
    all_images = []
    all_hyponyms = []
    for hypernym,path in filepaths.items():
        for opt_string in ["opt","nonopt"]:
            hyponyms_path= f"data/wordnet/{hypernym.replace(' ','_')}_{opt_string}.txt"
            with open(hyponyms_path,'r') as file:
                hyponyms_lst = file.read().split('\n')
                hyponyms_lst = [element for element in hyponyms_lst if (len(element) > 1)]
                all_hyponyms.extend(hyponyms_lst)
                hyponym[f"{hypernym}_{opt_string}"] = hyponyms_lst
        all_images.append(Image.open(path).convert("RGB"))
    similarities_wordnet = eval_clip(all_images, all_hyponyms,'ViT-L/14')
    print(similarities_wordnet)
    print(len(similarities_wordnet))
    row_idx = 0
    df_list = []
    # TODO: this is terrible, refactor by calling eval clip several times on subsets instead, unless there would be performance reasons
    for col_idx, hypernym in enumerate(filepaths.keys()):
        for opt_string in ["opt", "nonopt"]:
            for element in hyponym[f"{hypernym}_{opt_string}"]:
                df_list.append((hypernym,element,similarities_wordnet[row_idx,col_idx],opt_string))
                row_idx+=1
    print(df_list)
    df_swarm = pd.DataFrame(df_list, columns=["superclass","class","CLIP Score","optimized"])
    df_swarm['optimized'] = df_swarm["optimized"].str.replace("opt","optimized")
    df_swarm['optimized'] = df_swarm["optimized"].str.replace("non", "not ")
    df_swarm['targeted'] = df_swarm["optimized"].str.replace("optimized", "targeted")
    print(df_swarm)

    sns.set_context("paper")
    plt.figure()
    sns.swarmplot(data=df_swarm, x="superclass", y="CLIP Score", hue="targeted")
    plt.show()
    plt.savefig(os.path.join(FIGURE_OUTPUT_PATH,'figure_4.pdf'), format='pdf', dpi=300)

def plot_violines_extended_multi(similarities_imagenet,num_runs=10):
    fontsize = 7
    params = {  # 'backend': 'pdf',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial'}
    plt.rcParams.update(params)
    clip_scores_optimized = []
    clip_scores_imagenet = []

    samples = [25, 50, 75, 100, 250, 500, 750, 1000]
    from collections import defaultdict
    paths = defaultdict(list)

    # build paths dict
    for sample in samples:
        for run in range(num_runs):
            paths[sample].append((f"PGD", f'results/master_images/appendix/run{run}_sample_{sample}/cmpimagenet_pgd_int_run_{run}.png'))

    for sample in samples:
        for run in range(num_runs):
            opt_captions = sample_opt_captions(sample,random_seed=run)
            opt_captions = [caption for caption in opt_captions if caption]
            tokens_input = clip.tokenize(opt_captions).to(device)
            with torch.no_grad():
                text_features = clip_model_vit.encode_text(tokens_input).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            clip_scores_imagenet.extend(
            [(key, value, sample, f"ImageNet") for key, tensor_lst in similarities_imagenet.items() for value in tensor_lst if key in opt_captions])
            img_paths = [paths[sample][run][1]]
            print(img_paths)
            images = [Image.open(path).convert("RGB") for path in img_paths]
            images = [preprocess_vit(image) for image in images]
            image_input = torch.tensor(np.stack(images)).to(device)

            with torch.no_grad():
                image_features = clip_model_vit.encode_image(image_input).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity_single = text_features.cpu().numpy() @ image_features.cpu().numpy().T
            clip_scores_optimized.extend([(key, value[0], sample, f'{paths[sample][run][0]}') for key, value in zip(opt_captions, similarity_single.tolist())])

        # TODO: this needs to be transformed, we should only compute ImageNet similarities on demand once


    input_data = clip_scores_imagenet + clip_scores_optimized
    df = pd.DataFrame(input_data, columns=['class', 'CLIP score', 'number of optimized classes', 'type'])
    plt.figure()
    sns_plot = sns.catplot(#ci='sd',
        data=df, kind="violin",
        x='number of optimized classes', y="CLIP score", hue='type', height=2*2.26, aspect=1.416,
        palette=sns.color_palette('colorblind', n_colors=20))

    axes = sns_plot.axes.flatten()
    #plt.legend(loc='upper right')
    #plt.tight_layout()
    plt.show()
    sns_plot.savefig(os.path.join('figure_8.pdf'), dpi=300)
    df_x = df[df['type']=='PGD']

plot_art_heatmap()
model_dict = {}
# load clip model
clip_model_vit, preprocess_vit = clip.load('ViT-L/14', device=device)
clip_model_vit.eval()
model_dict['ViT-L/14'] = (clip_model_vit, preprocess_vit)
val_features, val_labels, captions, idx_to_class, val_features_unnorm = set_up_imagenet_features(clip_model_vit, preprocess_vit)
similarities_imagenet_vit = get_imagenet_similarities(clip_model_vit, val_features, val_labels, captions, idx_to_class)
json_object = json.dumps(similarities_imagenet_vit, indent=8)

with open("imagenet_scores.json", "w") as outfile:
    outfile.write(json_object)

plot_imagenet_scatter(clip_model_vit, preprocess_vit, similarities_imagenet_vit, sample_opt_captions(25))
plot_lines_multi(similarities_imagenet_vit)
compare_images("data/sunflower.jpg", 'results/master_images/cmp_imagenet_pgd_int_25.png', clip_model_vit, preprocess_vit)
generate_poi_table(val_features_unnorm, val_labels, captions, clip_model_vit, preprocess_vit)

clip_model_resnet, val_features, val_labels, captions, idx_to_class, val_features_unnorm = set_up_imagenet_features_custom('RN50x64', train=False, model_dict=model_dict)
similarities_imagenet_resnet = get_imagenet_similarities(clip_model_resnet, val_features, val_labels, captions, idx_to_class)

clip_model_blip, val_features, val_labels, captions, idx_to_class, val_features_unnorm = set_up_imagenet_features_custom('BLIP-384', train=False, model_dict=model_dict)
similarities_imagenet_blip = get_imagenet_similarities_blip(clip_model_blip, val_features, val_labels, captions, idx_to_class)

clip_model_siglip, val_features, val_labels, captions, idx_to_class, val_features_unnorm = set_up_imagenet_features_custom('ViT-L-16-SigLIP-384', train=False, model_dict=model_dict)
similarities_imagenet_siglip = get_imagenet_similarities_siglip(clip_model_siglip, val_features, val_labels, captions, idx_to_class)

plot_lines_multi_other_approaches(similarities_imagenet_vit, similarities_imagenet_resnet, similarities_imagenet_blip, similarities_imagenet_siglip)

wordnet_transfer()

# generated data is too large to be included in TMLR supplementary
#plot_violines_extended_multi(similarities_imagenet_vit)





