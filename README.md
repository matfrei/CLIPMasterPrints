# Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints

[![Paper](https://img.shields.io/badge/paper-arxiv.2307.03798-B31B1B.svg)](https://arxiv.org/abs/2307.03798) 

![alt text](static/demo.gif)

Installation
-------

clipmasterprints builds upon the stable diffusion conda enviroment and decoder model.
To run the code in the repository, you need to download and set up both:

```
mkdir external
cd external

# clone repository
git clone https://github.com/CompVis/stable-diffusion.git

# get correct commit
git checkout 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc

# created and activate conda env with SD dependencies
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm

# install SD from source into conda env
pip install -e .

# move previously downloaded SD sd-v1-4.ckpt into correct folder
# (Refer to https://github.com/CompVis/ for where to download the checkpoint)
ln -s <path/to/sd-v1-4.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 

# return to base dir
cd ../..

```



### Installing  BLIP depencies

To mine CLIPMasterPrints for BLIP, clone the BLIP github repository and add it to your PYTHONPATH

```
mkdir /path/to/blip
cd /path/to/blip
git clone https://github.com/salesforce/BLIP.git
cd BLIP
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/path/to/blip/BLIP
```


### Installing SigLip dependencies

All requirements for mining from SigLip models should be accounted for in the clipmasterprints package, open-clip-torch>=2.23.0 and timm>=0.9.8 are required

### Installing the CLIPMasterPrints package

After all Stable diffusion and BLIP dependencies are installed, install the package from source using

```
pip install -e .
```

Mining and evaluating CLIPMasterPrints
-------

To mine fooling master images, use
```
python train/mine.py --config-path config/<config-name>.yaml
```
where ```<config-name>``` is a placeholder for the desired config file.
Use ```cmp_artworks.yaml``` to target artwork captions or ```cmp_imagenet_classes_*.yaml``` to reproduce our experiments on imagenet class captions.

To display some plots for mined images, execute
```
python eval/eval_results.py
```
The shown plots are also saved under ```./figures/```

Training a classifier to detect CLIPMasterPrints
-------

### Mining the dataset
Since our used dataset is based on the ILSVRC 2012 imagenet subset, we cannot distribute it due to licensing reasons. You can mine a similar dataset though by calling

```
python tools/mine_adv_dataset.py
```
Do not forget to adjust the paths in the script accordingly to match the desired input and output paths on your file system,

After the dataset has been mined correctly and written to /path/to/dataset/tainted, assuming that the original ILRSVRC 2012 data can found under /path/to/imagenet, create a symbolic link to the imagenet folder into  /path/to/dataset/ in order to provide some negative examples:

```
ln -s /path/to/imagenet/  /path/to/dataset/purw
```

### Training a classifier
Again, after adjusting the path pointing to the adverserial dataset in the script, run

```
python train/train_adv_detector.py
```

### Evluating a trained classifier

```
python eval/eval_adv_detector.py
```


Authors
-------

Matthias Freiberger <mafr@di.ku.dk>

Peter Kun <peku@itu.dk>

Christian Igel <igel@di.ku.dk>

Anders Sundnes Løvlie <asun@itu.dk>
 
Sebastian Risi <sebr@itu.dk>

Citation
------
If you use the code for academic or commecial use, please cite the associated paper:
```
@misc{https://doi.org/10.48550/arXiv.2307.03798,
  doi = {10.48550/ARXIV.2307.03798},
  
  url = {https://arxiv.org/abs/2307.03798},
  
  author = {Freiberger, Matthias  and Kun, Peter and Igel, Christian and Løvlie, Anders Sundnes and Risi, Sebastian},
  
  title = {Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints},
  
  publisher = {arXiv},
  
  year = {2023},
}
