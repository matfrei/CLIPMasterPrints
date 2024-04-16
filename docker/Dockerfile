# Distributed under the terms of the TODO
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="Matthias Freiberger <mafr@di.ku.dk>"

# install miniconda
ARG PYTHON_VERSION=3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         rsync \
         htop \
         vim \
         wget \
         unzip \
         ca-certificates \
         libjpeg-dev \
         libgl1-mesa-glx \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN ls -la

RUN mkdir /external
WORKDIR /external

# clone SD repository
RUN git clone https://github.com/CompVis/stable-diffusion.git

WORKDIR /external/stable-diffusion

# get correct commit
RUN git checkout 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc

RUN conda init 

RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "ldm", "/bin/bash", "-c"]

EXPOSE 5003

# The code to run when container is started:
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python3", "src/server.py"]

#RUN conda init

#RUN conda activate ldm

# install SD from source into conda env
RUN pip install -e .

WORKDIR /

RUN git clone https://github.com/salesforce/BLIP.git
WORKDIR /BLIP
RUN pip install -r requirements.txt
ENV PYTHONPATH=$PYTHONPATH:/path/to/blip/BLIP

WORKDIR /
RUN git clone https://github.com/matfrei/CLIPMasterPrints.git

WORKDIR /CLIPMasterPrints
RUN ls -la
RUN pip install -e .



# Connect the dockerfile with it's original repo
LABEL org.opencontainers.image.source https://github.com/matfrei/CLIPMasterPrints
