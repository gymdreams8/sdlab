# sdlab
Stable Diffusion Lab using Python

## Installation

Developed on Apple Silicon, with pytorch nightly. See [GymDreams8 Docs Pytorch Nightly](https://docs.gymdreams8.com/mac_automatic1111.html#pytorch-nightly) for details.

```bash
# Create virtualenv with pyenv
pyenv virtualenv 3.10.9 sdlab

# Activate virtualenv
pyenv shell sdlab

# Install Pytorch Nightly
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install requirements
pip install -r requirements.txt

```

## Usage

### txt2img

Just run the script `./sdlab/txt2img.py` e.g.:

```sh
python ./sdlab/txt2img.py
```

## Notes

### CUDA 

I dev on an M2Max so the pipeline is intentionally not run on a GPU. I’ll add a config to run on CUDA later, but since this was just for me, having CUDA in this code is not a priority. I believe that all you need to add is:

```py
pipe.to("cuda")
```

### Folders

Also for my own dev only, but I save my images to Dropbox. You can replace these with your own paths and safe tensors models.
```py
txt2img = Text2Image(
    model_path='~/_StableDiffusion_Models/Stable-diffusion/airfucksBruteMix_v10.safetensors',
    outputs_path='~/Dropbox/StableDiffusion/outputs_code',
    prompt='man'
)
```

