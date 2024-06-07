# README #

## Update Dependencies ##
```commandline
git submodule update --init --recursive
```
## For DGX Station
1. add sudo group
```commandline
sudo usermod -aG sudo ${USER}
```
2. add docker group
```commandline
sudo usermod -aG docker ${USER}
```

## Install DGX based [pytorch docker](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) container
Pull Docker image
```commandline
docker pull nvcr.io/nvidia/pytorch:21.08-py3
```

START PYTHORCH DOCKER
```commandline
 docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:21.08-py3
```

This README would normally document whatever steps are necessary to get your application up and running.

## For Thai-SC Singularity container

1. load module
```commandline
module load Singularity
```
2. pull container
```commandline
singularity pull dgx_pytorch.sif docker://nvcr.io/nvidia/pytorch:21.08-py3
```
3. start container
```commandline
--for checking only @front-end node
singularity shell <container_name>

--run using SBATCH scripts
singularity exec --nv <singularity_name>.sif <command>       
#Run container --nv is for DGX node, In case of multiple-clusters use srun in front of the code
```

## install Pytorch (for any machine)
```commandline
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
