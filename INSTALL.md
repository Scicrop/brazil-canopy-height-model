## Installation and credentials

Here we present two ways how to install the packages. 
- A) requires GDAL to be installed on the system first. 
- B) GDAL is installed with mamba/conda in the environment.

### With pip in a virtual environment

1. Install [GDAL](https://gdal.org/). For Ubuntu follow e.g. these [instructions](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html).
2. Create a new [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) called `brchm` by running: `python3 -m venv $HOME/venvs/brchm`
3. Activate the environment:`source $HOME/venvs/brchm/bin/activate`. (Check that python points to the new environment with `which python3`.)
4. Install pytorch by following the instructions on [pytorch.org](https://pytorch.org/) that match your versions. Run e.g. `python3 -m pip install torch torchvision torchaudio`
5. pip install numpy==2.0.0rc2 wheel setuptools>=67
6. Install the GDAL python API matching the installed GDAL version: `python3 -m pip install GDAL==3.8.4`
7. pip install gdal[numpy]=="$(gdal-config --version).*" --no-build-isolation
8. python3 -c "import osgeo.gdal_array"
9. Install all other required packages: `python3 -m pip install -r requirements.txt`
10. Install this project as an editable package called `gchm`. Make sure you are in the directory of the repository containing the file `setup.py` . 
  Run: `python3 -m pip install -e .` (Note the dot `.` at the end.)

### Credentials for wandb
Optional. Only needed to run the training code (Not needed for deployment).
Create a file called `~/.config_wandb` containing your [weights and biases API key](https://docs.wandb.ai/quickstart): 
```
export WANDB_API_KEY=YOUR_API_KEY
```


### Credentials for AWS
Optional. This is only needed to download Sentinel-2 images from AWS on the fly using `gchm/deploy.py`. 
***Note that there are costs per GB downloaded!***

Create a file `~/.aws_configs` containing your AWS credentials as environment variables. 
```
export AWS_ACCESS_KEY_ID=PUT_YOUR_KEY_ID_HERE
export AWS_SECRET_ACCESS_KEY=PUT_YOUR_SECRET_ACCESS_KEY_HERE
export AWS_REQUEST_PAYER=requester
```
To create an AWS account go to: https://aws.amazon.com/console/
