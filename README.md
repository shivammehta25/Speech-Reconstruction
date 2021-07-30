# Speech Reconstruction via Generative Modelling

Uses Generative Modelling to reconsturct MelSpectrograms of images


## Set Up Environment
1. Open `environment.yml` set the `prefix` to your desired anaconda location
2. then run `conda env create -f environment.yml`



## Training

### VAE
1. open `src/hparams.py` set `model = "VAE"`
### GAN
1. open `src/hparams.py` set `model = "GAN"`
### DCGAN
1. open `src/hparams.py` set `model = "DCGAN"`


## Synthesising
1. `git clone https://github.com/NVIDIA/waveglow.git` Download WaveGlow Vocoder and place it in the same directory so that it is like `Speech-Reconstruction/waveglow`
2. open `jupyter-notebook` and open `generate_samples.ipyb` and follow the instructions there


# Some synthesised Examples

## Dataset Examples

![Dataset Sample 1](/images/dataset1.png)
![Dataset Sample 1](/images/dataset2.png)

## VAE
![Dataset Sample 1](/images/VAE1.png)
![Dataset Sample 1](/images/VAE2.png)

## GAN
![Dataset Sample 1](/images/GAN1.png)
![Dataset Sample 1](/images/GAN2.png)

## DCGAN
![Dataset Sample 1](/images/DCGAN1.png)
![Dataset Sample 1](/images/DCGAN2.png)

---

### Resources

VAE: https://github.com/TobiasNorlund/vq-vae
<br>
GANs: https://github.com/nocotan/pytorch-lightning-gans
