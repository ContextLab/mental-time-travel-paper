# Episodic memory: mental time travel or quantum 'memory wave' function?

This repository contains data and code used to produce the paper ["_Episodic memory: mental time travel or quantum 'memory wave' function?_"](https://psyarxiv.com/6zjwb) by Jeremy R. Manning. The repository is organized as follows:

```
root
└── code : all code used in the paper
    └── notebooks : jupyter notebooks for paper analyses
└── data : all data used in the paper
    ├── raw : raw data before processing
    └── processed : all processed data
└── paper : all files to generate paper
    └── figs : pdf copies of each figure
```

I also include a Dockerfile to reproduce our computational environment. Instruction for use are below (copied and modified from [this project](https://github.com/ContextLab/sherlock-topic-model-paper)):

## One time setup
1. Install Docker on your computer using the appropriate guide below:
    - [OSX](https://docs.docker.com/docker-for-mac/install/#download-docker-for-mac)
    - [Windows](https://docs.docker.com/docker-for-windows/install/)
    - [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
    - [Debian](https://docs.docker.com/engine/installation/linux/docker-ce/debian/)
2. Launch Docker and adjust the preferences to allocate sufficient resources (e.g. > 4GB RAM)
3. Build the docker image by opening a terminal in this repo folder and enter `docker build -t qwave .`  
4. Use the image to create a new container for the workshop
    - The command below will create a new container that will map your local copy of the repository to `/mnt` within the container, so that all the files in the repo are shared between your host OS and the container. The command will also share port `9999` with your host computer so any jupyter notebooks launched from *within* the container will be accessible at `localhost:9999` in your web browser
    - `docker run -it -p 9999:9999 --name qwave -v $PWD:/mnt qwave`
    - You should now see the `root@` prefix in your terminal, if so you've successfully created a container and are running a shell from *inside*!
5. To launch any of the notebooks: `jupyter notebook --port=9999 --no-browser --ip=0.0.0.0 --allow-root`

## Using the container after setup
1. You can always fire up the container by typing the following into a terminal
    - `docker start --attach qwave`
    - When you see the `root@` prefix, letting you know you're inside the container
2. Close a running container with `ctrl + d` from the same terminal you used to launch the container, or `docker stop qwave` from any other terminal
