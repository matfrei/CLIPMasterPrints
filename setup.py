from setuptools import find_packages, setup

setup(
    name='clipmasterprints',
    packages=find_packages(),
    version='0.0.2.2',
    description='Supplementary code for \"Fooling Contrastive Language-Image Pre-trained Models with CLIPMasterPrints\"',
    author='Matthias Freiberger',
    license='MIT',
    install_requires=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "numpy==1.23.3",
        "pandas==1.5.0",
        "cma==3.2.2",
        "yacs",
        "matplotlib==3.6.2",
        "seaborn==0.11",
        "open-clip-torch>=2.23.0",
        "timm==0.9.8",
        "transformers==4.19.2"
    ]
)

