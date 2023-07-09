import os
import json

from PIL import Image

import torch, torchvision
import clip
from pytorch_lightning import seed_everything
import numpy as np
from collections import OrderedDict

from clipmasterprints import clip_extract_image_embeddings_on_demand, get_similarities_per_class, scatter_optimized_classes,plot_similarity_heatmap
val_image_features_filename = 'features/embeddings/imagenet_val_clip_embeddings.pt'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def plot_art_heatmap():
    clip_model, preprocess = clip.load('ViT-L/14@336px',device=device)
    clip_model.eval()

    captions = open('data/famous_paintings.txt').read().split('\n') + ['Random noise image']
    img_filenames = ['Mona_Lisa.jpeg','last_supper.jpeg','starry_night.jpeg','scream.jpeg','Gernika.jpeg','the_kiss.jpeg','girl_with_a_pearl_earring.jpg','birth_of_venus.jpeg','las_meninas.jpeg','creation_of_adam.jpeg',]
    img_paths = [os.path.join('data/paintings',filename) for filename in img_filenames]
    img_paths += ['results/baseline/random.png','results/master_images/cmp_art.png']
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
    plot_similarity_heatmap(captions, images, similarity_art,'figure_1.pdf')



def plot_imagenet_scatter():
    #TODO: update path to correct imagenet validation set location here
    imagenet_path = '/home/common/datasets/imagenet2012/val'
    mapping_path = 'data/LOC_synset_mapping.txt'
    mapping_lst = open(mapping_path, 'r').read().split('\n')
    mapping = dict([(string_pair[:9], string_pair[9:].strip()) for string_pair in mapping_lst if string_pair])

    # load clip model
    clip_model, preprocess = clip.load('ViT-L/14',device=device)
    clip_model.eval()
    #get ImageNet validation set
    val_set = torchvision.datasets.ImageFolder(root=imagenet_path,transform=preprocess)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=3300, num_workers=7)

    idx_to_class = OrderedDict([(value, mapping[key]) for key, value in val_set.class_to_idx.items()])
    captions = list(idx_to_class.values())

    # get optimized captions
    seed_everything(0)
    caption_indices = np.random.permutation(len(captions))[:25].tolist()
    opt_captions = [captions[idx] for idx in caption_indices]
    print(opt_captions)

    tokens = clip.tokenize(captions).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    (val_features, val_labels) = clip_extract_image_embeddings_on_demand(clip_model, val_loader, val_image_features_filename,device = device)
    val_features /= val_features.norm(dim=-1,keepdim=True)

    similarities = get_similarities_per_class(val_features, val_labels, text_features, idx_to_class)
    similarities_imagenet = dict((key,[item for sublist in [element.cpu().numpy().tolist()[0] for element in value] for item in sublist])for key,value in similarities.items())

    # write imagenet scores out to json
    json_object = json.dumps(similarities_imagenet, indent=8)
    # to print the json_object and see the output
    
    with open("imagenet_scores.json", "w") as outfile:
        outfile.write(json_object)

    # now get scores for fooling image
    fooling_path = 'results/master_images/cmp_imagenet_25.png'
    opt_tokens = clip.tokenize(opt_captions).to(device)
    img_paths = [fooling_path]
    images = [Image.open(path).convert("RGB") for path in img_paths]
    images = [preprocess(image) for image in images]
    image_input = torch.tensor(np.stack(images)).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()
        text_features = clip_model.encode_text(opt_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarities_fooling = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    similarities_imagenet_opt = dict([(key, value) for key, value in similarities_imagenet.items() if key in opt_captions])
    scatter_optimized_classes(opt_captions,similarities_fooling, similarities_imagenet_opt,'figure_3.pdf')

plot_art_heatmap()
plot_imagenet_scatter()
