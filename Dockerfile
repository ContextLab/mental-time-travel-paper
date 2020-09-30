FROM contextlab/cdl-python:3.7

RUN conda install -c brainiak \
        brainiak=0.10 \
        matplotlib=3.2.2 \
        notebook=6.0.3 \
        pandas=1.0.5 \
        seaborn=0.10.1 \
        xlrd=1.2.0 \
    && conda clean -afy \
    && pip install hypertools==0.6.2 \
    && rm -rf ~/.cache/pip

# Set default working directory to repo mountpoint
WORKDIR /mnt

# Unset Python shell command from parent
ENTRYPOINT ["/usr/bin/env"]
CMD ["bash"]