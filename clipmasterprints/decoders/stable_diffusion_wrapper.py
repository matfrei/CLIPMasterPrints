import torch, torchvision
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# function adapted from stable diffusion demo code
# https://github.com/CompVis/stable-diffusion
def load_sd_model_from_config(config, ckpt, verbose=False):
    config = OmegaConf.load(config)
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model

class StableDiffusionWrapper(torch.nn.Module):

    def __init__(self, config, ckpt, image_dims):
        super().__init__()
        self.model = load_sd_model_from_config(config, ckpt, verbose=False)
        self.image_height = image_dims[0]
        self.image_width = image_dims[1]

        self.preprocessing = torchvision.transforms.Compose([torchvision.transforms.Resize(min([self.image_height, self.image_width])),
                                                       torchvision.transforms.CenterCrop(min([self.image_height, self.image_width])),
                                                       torchvision.transforms.Lambda(lambda x: 2 * x - 1)])
        self.postprocessing = torchvision.transforms.Lambda(lambda x: torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0))

    def encode(self, input):
        input_pp = self.preprocessing(input)
        return self.model.get_first_stage_encoding(self.model.encode_first_stage(input_pp))

    def decode(self, input):
        output = self.model.decode_first_stage(input)
        return self.postprocessing(output)
class StableDiffusionWrapperWGradient(StableDiffusionWrapper):
    def decode(self, input):
        input = 1. / self.model.scale_factor * input
        return self.postprocessing(self.model.first_stage_model.decode(input))