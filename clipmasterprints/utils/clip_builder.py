import torch, torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def build_clip(clip_string,device,tensor_input=True):
    import clip
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
    preprocess =(preprocess if not tensor_input else torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(clip_input_res), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
    torchvision.transforms.CenterCrop(size=(clip_input_res,clip_input_res)),
    torchvision.transforms.Normalize(mean=input_mean, std=input_std)]))

    clip_model.eval()
    clip_model.to(device)
    return (clip_string,clip_model,preprocess)

#FIXXXME: unify all builderfunction into a single builderfunction calling the specialist builder function
#         (factory method pattern)
def build_blip(blip_string,device,tensor_input=True):
    from models.blip_itm import blip_itm
    image_size = 384
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base',med_config='../BLIP/configs/med_config.json')
    model.eval()
    model = model.to(device=device)

    preprocess_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    preprocess = (preprocess_tensor if tensor_input else transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]))

    return (blip_string,model,preprocess)

def build_siglip(siglip_string,device,tensor_input=True):
    from open_clip import create_model_from_pretrained  # works on open-clip-torch>=2.23.0, timm>=0.9.8
    image_size = 384
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-384')
    model.eval()
    model = model.to(device=device)
    preprocess_tensor = transforms.Compose(#[preprocess.transforms[0],preprocess.transforms[3]])
        [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
        preprocess.transforms[3]])
    preprocess = (preprocess if not tensor_input else preprocess_tensor)

    return (siglip_string,model,preprocess)