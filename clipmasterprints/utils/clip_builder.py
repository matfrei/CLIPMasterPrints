import clip
import torch,torchvision

def build_clip(clip_string,device):
    try:
        clip_model, preprocess = clip.load(clip_string,device='cpu')
    except EOFError:
        clip_model, preprocess = clip.load('ViT-L/14',device='cpu')
        state_dict = torch.load(clip_string)
        clip_model.load_state_dict(state_dict['state_dict'])

    clip_input_res = clip_model.visual.input_resolution
    input_mean = preprocess.transforms[-1].mean
    input_std = preprocess.transforms[-1].std

    # original preprocess function assumes a PIL image while we work already with tensors, so use a custom preprocessing funcion
    tensor_preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(clip_input_res,clip_input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    torchvision.transforms.CenterCrop(size=(clip_input_res,clip_input_res)),
    torchvision.transforms.Normalize(mean=input_mean, std=input_std)])

    clip_model.eval()
    clip_model.to(device)
    return (clip_string,clip_model,tensor_preprocessing)