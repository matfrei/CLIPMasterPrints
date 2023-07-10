from yacs.config import CfgNode as CN

_C = CN()

# Data
_C.DATA = CN()
_C.DATA.CAPTION_PATH = '../data/famous_paintings.txt'
_C.DATA.IMAGENET_TRAIN = '/home/common/datasets/imagenet2012/train/'
_C.DATA.IMAGENET_TEST = '/home/common/datasets/imagenet2012/val/'
_C.DATA.BS_IMAGENET_EVAL = 3200

_C.FEATURES = CN()
_C.FEATURES.PATH_TRAIN = 'features/imagenet_train_clip_embeddings.pt'
_C.FEATURES.PATH_TEST = 'features/imagenet_val_clip_embeddings.pt'

# Experiment parameters
_C.RANDOM_SEED = 0
_C.NUM_RUNS = 1
_C.SAMPLE_CAPTIONS = ['all']
_C.GPU_ID = 0

# Experiment logging information
_C.EXPERIMENT_LOG = CN()
_C.EXPERIMENT_LOG.BASEPATH = 'results-reproduced/CLIPMasterPrints'
_C.EXPERIMENT_LOG.MODEL_NAME = 'CMA-ES+SDDecode'
_C.EXPERIMENT_LOG.EXPERIMENT_NAME = 'mine-clipmasterprint'

# Loss class to use
_C.LOSS = CN()
_C.LOSS.CLASS = "CLIPLoss"

# parameters specific to autoencoder
_C.AUTOENCODER = CN()
_C.AUTOENCODER.CLASS_NAME = 'StableDiffusionWrapper'
_C.AUTOENCODER.CONFIG_PATH = 'external/stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
_C.AUTOENCODER.WEIGHT_PATH = 'external/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt'
_C.AUTOENCODER.IMG_HEIGHT = 336
_C.AUTOENCODER.IMG_WIDTH = 336
_C.AUTOENCODER.LATENT_CHANNELS = 4
_C.AUTOENCODER.DOWNSAMPLING_FACTOR = 8
_C.AUTOENCODER.BATCH_SIZE = 15

# parameters specific to CLIP
_C.CLIP = CN()
_C.CLIP.MODEL_STRINGS = ['ViT-L/14@336px']
_C.CLIP.BATCH_SIZE = 15
_C.CLIP.SHIFT = 0.
# optimizer to use
_C.OPTIMIZER = CN()
_C.OPTIMIZER.METHOD = 'CMA-ES'
_C.OPTIMIZER.ITER = 20000
_C.OPTIMIZER.CHECK_POINT_AFTER_X_ITER = 300
_C.OPTIMIZER.BATCH_SIZE = 14
_C.OPTIMIZER.LR = 5e-5

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for foolingclip"""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
