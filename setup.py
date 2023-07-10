from setuptools import find_packages, setup

setup(
    name='clipmasterprints',
    packages=find_packages(),
    version='0.0.1',
    description='Code base for CLIPMasterPrints: Fooling Contrastive Language-Image Pre-training Using Latent Variable Evolution',
    author='Matthias Freiberger',
    license='',
    install_requires=[
        "torch==1.11.0",
        "torchvision==0.12.0",
        "numpy==1.23.3",
        "pandas==1.5.0",
        "cma==3.2.2",
        "yacs",
        "matplotlib==3.6.2",
        "seaborn==0.12.1"

    ]
)

