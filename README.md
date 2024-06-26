# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

Unofficial PyTorch implementation of [Vocos](https://openreview.net/forum?id=vY9nzQmQBw) by Hubert Siuzdak.

Vocos is a fast neural vocoder designed to synthesize audio waveforms from acoustic features. Trained using a Generative Adversarial Network (GAN) objective, Vocos can generate waveforms in a single forward pass. Unlike other typical GAN-based vocoders, Vocos does not model audio samples in the time domain. Instead, it generates spectral coefficients, facilitating rapid audio reconstruction through inverse Fourier transform.

Audio samples and some information are provided in the [web](https://gemelo-ai.github.io/vocos/).

<center><img src="model-voco.png"></center>

## Installation
Clone the repository and install dependencies.
```shell
# the codebase has been tested on Python 3.10 with PyTorch 2.0.1 binaries
git clone https://github.com/LqNoob/vocos
pip install -r requirements.txt
```

## Training

Prepare a filelist of audio files for the training and validation set. 
Fill a config file with your filelist paths and start training with:

```bash
python train.py -c configs/vocos-fs2.yaml
```

If you need to do fine-tune on the model. Run the following command：
```bash
python train.py -c configs/vocos-fs2-gta.yaml
```

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.

## Inference (copy synthesis)

Inference only supports the copy synthesis process, see:
```bash
python inference.py
```

## Acknowledgements
We referred to [vocos](https://github.com/gemelo-ai/vocos) and [matcha](https://github.com/wetdog/vocos/tree/matcha) to implement this.
