# hash:sha256:cf58b5829b621673f38b83a30c7882964c32cab7599bd52a02c652383d17e95a
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y --channel=rdkit --channel=pytorch --channel=conda-forge \
        dask==2.16.0 \
        ipywidgets==7.5.1 \
        isort==4.3.21 \
        jupyter==1.0.0 \
        jupyter_contrib_nbextensions==0.5.1 \
        matplotlib==3.1.3 \
        nb_conda_kernels==2.2.3 \
        numpy==1.18.4 \
        openpyxl==3.0.3 \
        pytorch==1.6.0 \
        rdkit==2020.03.2.0 \
        scikit-learn==0.22.1 \
        scipy==1.4.1 \
        seaborn==0.10.1 \
        tqdm==4.48.2 \
        xlrd==1.2.0 \
        yapf==0.29.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    more-itertools==8.2.0 \
    nujson==1.35.2 \
    selfies==0.2.4
