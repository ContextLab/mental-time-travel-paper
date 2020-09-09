FROM debian:stretch

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG C.UTF-8
ENV PATH /opt/conda/bin:$PATH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        vim \
    && rm -rf /var/lib/apt/lists/* \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && conda update setuptools \
    && conda install -c brainiak -c defaults -c conda-forge --strict-channel-priority \
        brainiak=0.10 \
        matplotlib=3.2.2 \
        notebook=6.0.3 \
        pandas=1.0.5 \
        seaborn=0.10.1 \
    && conda clean --all -f -y \
    && pip install --no-cache-dir hypertools==0.6.2

# Set default working directory to repo mountpoint
WORKDIR /mnt
