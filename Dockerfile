FROM debian:stretch
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends eatmydata \
    && eatmydata apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mpich \
        procps \
        sudo \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && conda config --append channels conda-forge \
    && conda config --set auto_update_conda false \
    && conda config --set channel_priority strict \
    && conda update -S setuptools \
    && conda install -Sy \
        libgcc-ng \
        ipython=7.16.1 \
        ipywidgets=7.5.1 \
        tqdm=4.47.0 \
        pandoc=2.10 \
        nbconvert=5.6.1 \
        jupyterlab=2.1.5 \
        bokeh=2.1.1 \
        h5py=2.10.0 \
        joblib=0.16.0 \
        matplotlib=3.2.2 \
        numba=0.50.1 \
        numexpr=2.7.1 \
        numpy=1.19.1 \
        openpyxl=3.0.4 \
        pandas=1.0.5 \
        pandas-profiling=1.4.1 \
        plotly=4.8.2 \
        pymysql=0.9.3 \
        pynndescent=0.4.8 \
        python-dateutil=2.8.1 \
        requests=2.24.0 \
        requests-ftp=0.3.1 \
        requests-kerberos=0.12.0 \
        scikit-image=0.16.2 \
        scikit-learn=0.23.1 \
        scipy=1.5.0 \
        seaborn=0.10.1 \
        sqlalchemy=1.3.18 \
        sqlalchemy-utils=0.36.5 \
        umap-learn=0.4.6 \
        urllib3=1.25.9 \
        xlrd=1.2.0 \
    && conda clean --all -f -y
RUN git clone https://github.com/FFmpeg/FFmpeg \
    && cd FFmpeg \
    && ./configure --enable-gpl \
    && make && make install \
    && ldconfig \
    && cd .. \
    && rm -r FFmpeg
RUN pip install --no-cache-dir \
        git+https://github.com/ContextLab/hypertools.git@dea0df127b150c39ed9a7b1faaa8fbde089ee834 \
        git+https://github.com/brainiak/brainiak.git@v0.7.1
WORKDIR mnt/