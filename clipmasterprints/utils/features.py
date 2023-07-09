import os
from tqdm import tqdm
import torch
from pathlib import Path
from collections import defaultdict

def clip_extract_image_embeddings(model, dataloader, device):
    features_lst = []
    labels_lst = []
    for batch in tqdm(dataloader):
        images = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
        features_lst.append(image_features)
        labels_lst.append(labels)
    return (torch.cat(features_lst,dim=0),torch.cat(labels_lst,dim=0))

def clip_extract_image_embeddings_on_demand(model,dataloader, path, device):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        (image_features, labels) = clip_extract_image_embeddings(model, dataloader, device)
        with open(path,'wb') as file:
            torch.save((image_features,labels),file)

    with open(path,'rb') as file:
        (image_features, labels) = torch.load(file,map_location=device)
    return(image_features,labels)

def eval_accuracy(image_features, text_features, labels):
    image_features/=image_features.norm(dim=1,keepdim=True)
    text_features/=text_features.norm(dim=1,keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    predicted_class = torch.argmax(text_probs, dim=1)
    return torch.sum(predicted_class == labels).cpu().numpy()/image_features.shape[0]

def get_similarities_per_class(image_features, image_labels, text_features, idx_to_class):
    similarities = defaultdict(list)
    classes = torch.unique(image_labels)
    for imagenet_class in classes:
        caption = idx_to_class[imagenet_class.item()]
        similarities[caption].append(
        torch.mm(text_features[imagenet_class].view(1, -1), image_features[image_labels == imagenet_class].T))
    return similarities

def eval_fooling_accuracy(image_features, image_labels, text_features, fooling_vec, classes):
    elements_total = 0
    elements_fooled = 0
    for imagenet_class in classes:
        text_vec = text_features[imagenet_class].view(1, -1)
        image_vecs = image_features[image_labels == imagenet_class]
        result = torch.mm(text_vec,image_vecs.T)
        fooling_result = text_vec @ fooling_vec.T
        elements_total += result.flatten().shape[0]
        elements_fooled += torch.sum(torch.ones_like(result)[result < fooling_result])

    return elements_fooled/elements_total

def get_others_per_class(image_features, image_labels, text_features, idx_to_class):
    similarities = defaultdict(list)
    classes = torch.unique(image_labels)
    for imagenet_class in classes:
        caption = idx_to_class[imagenet_class.item()]
        similarities[caption].append(
        torch.mm(text_features[imagenet_class].view(1, -1), image_features[image_labels != imagenet_class].T))
    return similarities